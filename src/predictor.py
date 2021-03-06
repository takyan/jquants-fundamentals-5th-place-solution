# -*- coding: utf-8 -*-
import io
import os
import pickle

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


AGG_FEATURE = [
    'volatility_3month', 
    'return_3month', 
    'MA_gap_3month',
    'per', 
    'pbr', 
    'roe', 
    'roa', 
    'profit_margin', 
    'equity_ratio', 
    'total_asset_turnover', 
    'receivables_turnover', 
    'dividend_yield', 
    'dividend_payout_ratio', 
    'eps', 
    'Result_FinancialStatement NetSales', 
    'Result_FinancialStatement OperatingIncome', 
    'Result_FinancialStatement OrdinaryIncome', 
    'EndOfDayQuote ExchangeOfficialClose'
]

PREV_YEAR_FEATURE = [
'Result_FinancialStatement NetSales',
 'Result_FinancialStatement OperatingIncome',
 'Result_FinancialStatement OrdinaryIncome',
 'Result_FinancialStatement NetIncome',
 'Result_FinancialStatement TotalAssets',
 'Result_FinancialStatement NetAssets',
 'Result_FinancialStatement CashFlowsFromOperatingActivities',
 'Result_FinancialStatement CashFlowsFromFinancingActivities',
 'Result_FinancialStatement CashFlowsFromInvestingActivities',
 'Forecast_FinancialStatement NetSales',
 'Forecast_FinancialStatement OperatingIncome',
 'Forecast_FinancialStatement OrdinaryIncome',
 'Forecast_FinancialStatement NetIncome'
]

QUARTER_FEATURE = [
    'Result_FinancialStatement NetSales', 
    'Result_FinancialStatement OperatingIncome', 
    'Result_FinancialStatement OrdinaryIncome', 
    'Result_FinancialStatement NetIncome'
]

UNUSED_FEATURE = [
    'Forecast_Dividend FiscalYear', 
    'Forecast_Dividend FiscalPeriodEnd', 
    'Forecast_Dividend ModifyDate', 
    'Forecast_Dividend RecordDate', 
    'Result_Dividend ModifyDate', 
    'Result_Dividend RecordDate', 
    'Result_Dividend DividendPayableDate', 
    'DividendPayableDate', 
    'Forecast_FinancialStatement FiscalYear', 
    'Forecast_FinancialStatement ModifyDate', 
    'Forecast_FinancialStatement FiscalPeriodEnd', 
    'base_date', 
    'Local Code', 
    'code', 
    'Result_Dividend FiscalPeriodEnd', 
    'IssuedShareEquityQuote AccountingStandard', 
    'IssuedShareEquityQuote ModifyDate', 
    "33 Sector(Code)", 
    "17 Sector(Code)", 
    'prediction_target', 
    'Effective Date', 
    'Name (English)', 
    'Size Code (New Index Series)', 
    'Result_FinancialStatement FiscalPeriodEnd', 
    'Result_FinancialStatement ModifyDate', 
    'Result_FinancialStatement FiscalYear', 
    'Result_Dividend FiscalYear', 
    'prev_year_Forecast_Dividend FiscalYear', 
    'prev_year_Forecast_Dividend FiscalPeriodEnd', 
    'prev_year_Forecast_Dividend ModifyDate', 
    'prev_year_Forecast_Dividend RecordDate', 
    'prev_year_Result_Dividend ModifyDate', 
    'prev_year_Result_Dividend RecordDate', 
    'prev_year_Result_Dividend DividendPayableDate', 
    'prev_year_DividendPayableDate', 
    'prev_year_Forecast_FinancialStatement FiscalYear', 
    'prev_year_Forecast_FinancialStatement ModifyDate', 
    'prev_year_Forecast_FinancialStatement FiscalPeriodEnd', 
    'prev_year_base_date', 
    'prev_year_Local Code', 
    'prev_year_code', 
    'prev_year_Result_Dividend FiscalPeriodEnd', 
    'prev_year_IssuedShareEquityQuote AccountingStandard', 
    'prev_year_IssuedShareEquityQuote ModifyDate', 
    'prev_year_Result_FinancialStatement FiscalPeriodEnd', 
    'prev_year_Result_FinancialStatement ModifyDate', 
    'prev_year_Result_FinancialStatement FiscalYear', 
    'Result_Dividend FiscalYear', 
    'Forecast_FinancialStatement AccountingStandard', 
    'Forecast_FinancialStatement CompanyType', 
    'prev_year_Forecast_FinancialStatement AccountingStandard', 
    'prev_year_Forecast_FinancialStatement CompanyType', 
    "prev_year_Forecast_FinancialStatement AccountingStandard", 
    "prev_year_Forecast_FinancialStatement ReportType", 
    "prev_year_Forecast_FinancialStatement CompanyType", 
    "prev_year_Result_Dividend FiscalYear", 
    'prev_year_Unnamed: 0', 
    'Unnamed: 0',
    'Section/Products'
]

def calc_OBV(df):
    return np.where(df['EndOfDayQuote ExchangeOfficialClose'] > df['EndOfDayQuote ExchangeOfficialClose'].shift(1), df['EndOfDayQuote Volume'], np.where(df['EndOfDayQuote ExchangeOfficialClose'] < df['EndOfDayQuote ExchangeOfficialClose'].shift(1), -df['EndOfDayQuote Volume'], 0)).cumsum()

def calc_BOP(df):
    return (df['EndOfDayQuote ExchangeOfficialClose'] - df['EndOfDayQuote Open']) / (df['EndOfDayQuote High'] - df['EndOfDayQuote Low'])

def calc_PPO(df, slow_ma_days=12, fast_ma_days=26):
    slow_ma = df["EndOfDayQuote ExchangeOfficialClose"].rolling(slow_ma_days).mean()
    fast_ma = df["EndOfDayQuote ExchangeOfficialClose"].rolling(fast_ma_days).mean()
    return ((fast_ma - slow_ma) / slow_ma) * 100

def calc_WILLR(df, days=14):
    df_high_rate = df['EndOfDayQuote High']
    df_low_rate = df['EndOfDayQuote Low']
    
    df_high = df_high_rate.rolling(days).max()
    df_low = df_low_rate.rolling(days).min()
    
    return (df['EndOfDayQuote ExchangeOfficialClose'] - df_high)/(df_high - df_low) * 100

def calc_DI_and_ADX_NATR(df, days=14):
    #DMI?????????
    df_pDM = (df['EndOfDayQuote High'] - df['EndOfDayQuote High'].shift(1))
    df_mDM = (df['EndOfDayQuote Low'].shift(1) - df['EndOfDayQuote Low'])
    
    df_pDM.loc[df_pDM<0] = 0
    df_pDM.loc[df_pDM-df_mDM < 0] = 0
    
    df_mDM.loc[df_mDM<0] = 0
    df_mDM.loc[df_mDM-df_pDM < 0] = 0
    
    #tr?????????
    a = (df['EndOfDayQuote High'] - df['EndOfDayQuote Low']).abs()
    b = (df['EndOfDayQuote High'] - df['EndOfDayQuote ExchangeOfficialClose'].shift(1)).abs()
    c = (df['EndOfDayQuote Low'] - df['EndOfDayQuote ExchangeOfficialClose'].shift(1)).abs()
    
    df_tr = pd.concat([a, b, c], axis=1).max(axis=1)
    
    df[f'pDI_{days}'] = df_pDM.rolling(days).sum()/df_tr.rolling(days).sum() * 100
    df[f'mDI_{days}'] = df_mDM.rolling(days).sum()/df_tr.rolling(days).sum() * 100
    
    df_DX = (df[f'pDI_{days}']-df[f'mDI_{days}']).abs()/(df[f'pDI_{days}']+df[f'mDI_{days}']) * 100
    df_DX = df_DX.fillna(0)
    df[f'ADX_{days}'] = df_DX.rolling(days).mean()

    df_tr_ema = df_tr.ewm(span=days).mean()
    df_atr = df_tr_ema
    df[f'NATR_{days}'] =df_atr / df['EndOfDayQuote ExchangeOfficialClose'] * 100
    
    return df

def hand_crafted_ta(df, days=14):
    df['OBV'] = calc_OBV(df)
    df['BOP'] = calc_BOP(df)
    df['PPO'] = calc_PPO(df)
    df[f'WILLR_{days}'] = calc_WILLR(df, days=days)
    df = calc_DI_and_ADX_NATR(df, days=days)
    return df

def create_stock_indicator(df):
    df["market_cap"] = df["EndOfDayQuote ExchangeOfficialClose"]*df["IssuedShareEquityQuote IssuedShare"]
    df["per"] = df["EndOfDayQuote ExchangeOfficialClose"]/(df["Result_FinancialStatement NetIncome"]*1000000/df["IssuedShareEquityQuote IssuedShare"])
    df["per"][df["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = np.nan
    df["per"][np.isinf(df["per"])] = np.nan
    df["pbr"] = df["EndOfDayQuote ExchangeOfficialClose"]/(df["Result_FinancialStatement NetAssets"]*1000000/df["IssuedShareEquityQuote IssuedShare"])
    df["pbr"][np.isinf(df["pbr"])] = np.nan
    df["roe"] = df["pbr"]/df["per"]

    df["roa"] = df["Result_FinancialStatement NetIncome"]/df['Result_FinancialStatement TotalAssets']
    df["roa"][np.isinf(df["roa"])] = np.nan

    df["profit_margin"] = df["Result_FinancialStatement NetIncome"]/df["Result_FinancialStatement NetSales"]
    df["profit_margin"][df["Result_FinancialStatement CashFlowsFromOperatingActivities"] == 0] = np.nan
    df["profit_margin"][np.isinf(df["profit_margin"])] = np.nan
    df["equity_ratio"] = df["Result_FinancialStatement NetAssets"]/df["Result_FinancialStatement TotalAssets"]

    df["total_asset_turnover"] = df["Result_FinancialStatement NetSales"] / df['Result_FinancialStatement TotalAssets']
    df["total_asset_turnover"][np.isinf(df["total_asset_turnover"])] = np.nan

    df["receivables_turnover"] = df["Result_FinancialStatement NetSales"] /( df['Result_FinancialStatement TotalAssets'] - df['Result_FinancialStatement NetAssets'])
    df["receivables_turnover"][np.isinf(df["receivables_turnover"])] = np.nan

    df["dividend_yield"] = df["Result_Dividend QuarterlyDividendPerShare"] / df["EndOfDayQuote ExchangeOfficialClose"]
    df["dividend_yield"][np.isinf(df["dividend_yield"])] = np.nan

    df["dividend_payout_ratio"] = (df["Result_Dividend QuarterlyDividendPerShare"] * df["IssuedShareEquityQuote IssuedShare"]) / df["Result_FinancialStatement NetIncome"]
    df["dividend_payout_ratio"][np.isinf(df["dividend_payout_ratio"])] = np.nan

    df["eps"] = df["Result_FinancialStatement NetIncome"] / df["IssuedShareEquityQuote IssuedShare"]
    df["eps"][np.isinf(df["eps"])] = np.nan

    return df

def missing_label_treatment(df, le, label):
    df.loc[df[label].isin(set(df[label]) - set(le.classes_)), label] = np.nan

def label_encode(df, target_column, model_path="../model"):

    m = os.path.join(model_path, f"le_{target_column}.pkl")
    with open(m, "rb") as f:
        le = pickle.load(f)

    missing_label_treatment(df, le, target_column)
    df.loc[df[target_column].isnull() == False, target_column] = le.transform(df[df[target_column].isnull() == False][target_column])
    df[target_column] = df[target_column].astype(float)

    return df

class ScoringService(object):
    # ?????????????????????
    TRAIN_END = "2019-12-31"
    # ????????????????????????
    TEST_START = "2020-01-01"
    # ????????????
    TARGET_LABELS = ["label_high_20", "label_low_20"]

    # ???????????????????????????????????????
    dfs = None
    # ???????????????????????????????????????
    models = None
    # ??????????????????????????????????????????????????????
    codes = None

    @classmethod
    def get_inputs(cls, dataset_dir):
        """
        Args:
            dataset_dir (str)  : path to dataset directory
        Returns:
            dict[str]: path to dataset files
        """
        inputs = {
            "stock_list": f"{dataset_dir}/stock_list.csv.gz",
            "stock_price": f"{dataset_dir}/stock_price.csv.gz",
            "stock_fin": f"{dataset_dir}/stock_fin.csv.gz",
            "stock_labels": f"{dataset_dir}/stock_labels.csv.gz",
        }
        return inputs

    @classmethod
    def get_dataset(cls, inputs):
        """
        Args:
            inputs (list[str]): path to dataset files
        Returns:
            dict[pd.DataFrame]: loaded data
        """
        if cls.dfs is None:
            cls.dfs = {}
        for k, v in inputs.items():
            if k != 'stock_fin_price':
                cls.dfs[k] = pd.read_csv(v)
                # DataFrame???index?????????????????????
                if k == "stock_price":
                    cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                        cls.dfs[k].loc[:, "EndOfDayQuote Date"]
                    )
                    cls.dfs[k].set_index("datetime", inplace=True)
                elif k in ["stock_fin", "stock_labels"]:
                    cls.dfs[k].loc[:, "datetime"] = pd.to_datetime(
                        cls.dfs[k].loc[:, "base_date"]
                    )
                    cls.dfs[k].set_index("datetime", inplace=True)
        return cls.dfs

    @classmethod
    def get_codes(cls, dfs):
        """
        Args:
            dfs (dict[pd.DataFrame]): loaded data
        Returns:
            array: list of stock codes
        """
        stock_list = dfs["stock_list"].copy()
        # ???????????????????????????????????????
        cls.codes = stock_list[stock_list["prediction_target"] == True][
            "Local Code"
        ].values
        return cls.codes

    @classmethod
    def get_features_for_predict(cls, dfs, code, start_dt="2016-01-01"):
        """
        Args:
            dfs (dict)  : dict of pd.DataFrame include stock_fin, stock_price
            code (int)  : A local code for a listed company
            start_dt (str): specify date range
        Returns:
            feature DataFrame (pd.DataFrame)
        """
        # stock_fin????????????????????????
        stock_fin = dfs["stock_fin"].copy()

        # ?????????????????????????????????????????????
        fin_data = stock_fin[stock_fin["Local Code"] == code].copy()
            
        temp_dfs = []
        for quarter in ['Annual', 'Q1', 'Q2', 'Q3']:
            temp = fin_data[fin_data['Result_FinancialStatement ReportType'] == quarter].drop_duplicates('Result_FinancialStatement FiscalPeriodEnd', keep='last').shift().add_prefix('prev_year_')
            temp_dfs.append(temp)
        df_shift = pd.concat(temp_dfs)
        fin_data = pd.concat([fin_data, df_shift], axis=1)

        target_columns = ['Result_FinancialStatement NetSales', 'Result_FinancialStatement OperatingIncome',
                          'Result_FinancialStatement OrdinaryIncome', 'Result_FinancialStatement NetIncome']

        temp_fin_data = fin_data.drop_duplicates('Result_FinancialStatement FiscalPeriodEnd', keep='last').copy()
        fin_data = pd.concat([fin_data, (temp_fin_data[target_columns] - temp_fin_data[target_columns].shift()).add_prefix("quarter_")], axis=1)
        fin_data.loc[fin_data['Result_FinancialStatement ReportType'] == 'Q1', [f"quarter_{column}" for column in target_columns]] = fin_data[fin_data['Result_FinancialStatement ReportType'] == 'Q1'][target_columns].values

        # ????????????pd.Timestamp??????????????????index?????????
        fin_data["datetime"] = pd.to_datetime(fin_data["base_date"])
        fin_data.set_index("datetime", inplace=True)
            
        # ??????????????????????????????60???????????????????????????????????????????????????
        # ???????????????????????????????????????????????????????????????90???????????????????????????????????????????????????

        # stock_price????????????????????????
        price = dfs["stock_price"].copy()
        # ?????????????????????????????????????????????
        price_data = price[price["Local Code"] == code].copy()
        # ????????????pd.Timestamp??????????????????index?????????
        price_data["datetime"] = pd.to_datetime(price_data["EndOfDayQuote Date"])
        price_data.set_index("datetime", inplace=True)

        feats = price_data[["EndOfDayQuote Open", "EndOfDayQuote High", "EndOfDayQuote Low", "EndOfDayQuote ExchangeOfficialClose", 'EndOfDayQuote Volume', 'EndOfDayQuote PreviousExchangeOfficialClose']].copy()
        # ???????????????????????????????????????

        # ?????????60?????????????????????
        feats["return_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"].pct_change(60)
        # ?????????60??????????????????????????????
        feats["volatility_3month"] = (
            np.log(feats["EndOfDayQuote ExchangeOfficialClose"]).diff().rolling(60).std()
            )
        # ?????????60??????????????????????????????????????????
        feats["MA_gap_3month"] = feats["EndOfDayQuote ExchangeOfficialClose"] / (
            feats["EndOfDayQuote ExchangeOfficialClose"].rolling(60).mean()
            )

        #Feature Engineering ta
        feats = hand_crafted_ta(feats)
            
        # ?????????????????????????????????
        feats = feats.drop(["EndOfDayQuote Open", "EndOfDayQuote High", "EndOfDayQuote Low", 'EndOfDayQuote Volume'], axis=1)

        # ??????????????????????????????????????????????????????????????????????????????????????????????????????
        n = 90
        # ???????????????????????????????????????
        fin_feats = fin_data.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]
        feats = feats.loc[pd.Timestamp(start_dt) - pd.offsets.BDay(n) :]

        feats = feats.loc[feats.index.isin(fin_feats.index)]
        fin_feats = fin_feats.loc[fin_feats.index.isin(feats.index)]

        # ??????????????????
        feats = pd.concat([feats, fin_feats], axis=1)#.dropna()

        # ????????????????????????
        feats["code"] = code

        # ??????????????????????????????????????????
        feats = feats.loc[pd.Timestamp(start_dt) :]
        
        return feats

    @classmethod
    def get_model(cls, model_path="../model", labels=None):
        """Get model method

        Args:
            model_path (str): Path to the trained model directory.
            labels (arrayt): list of prediction target labels

        Returns:
            bool: The return value. True for success, False otherwise.

        """
        if cls.models is None:
            cls.models = {}
        if labels is None:
            labels = cls.TARGET_LABELS
        try:
            for label in labels:
                m = os.path.join(model_path, f"my_model_{label}.pkl")
                with open(m, "rb") as f:
                    # pickle??????????????????????????????????????????????????????
                    cls.models[label] = pickle.load(f)
            return True
        except Exception as e:
            print(e)
            return False

    @classmethod
    def predict(cls, inputs, labels=None, codes=None, start_dt=TEST_START, model_path="../model"):
        """Predict method

        Args:
            inputs (dict[str]): paths to the dataset files
            labels (list[str]): target label names
            codes (list[int]): traget codes
            start_dt (str): specify date range
        Returns:
            str: Inference for the given input.
        """

        # ?????????????????????
        if cls.dfs is None:
            cls.get_dataset(inputs)
            cls.get_codes(cls.dfs)

        # ??????????????????????????????????????????????????????
        if codes is None:
            codes = cls.codes
        if labels is None:
            labels = cls.TARGET_LABELS

        # ??????????????????
        buff = []
        for code in tqdm(codes):
            buff.append(cls.get_features_for_predict(cls.dfs, code, start_dt))
        feats = pd.concat(buff).reset_index()

        feats = feats.merge(cls.dfs["stock_list"], on='Local Code', how='left')
        feats = create_stock_indicator(feats)
        
        df_groupby = feats.groupby(["Result_FinancialStatement FiscalYear", "Result_FinancialStatement ReportType", "33 Sector(name)"]).mean()[AGG_FEATURE].add_prefix('quaterly_agg_33sec_')
        feats = feats.merge(df_groupby, on=["Result_FinancialStatement FiscalYear", "Result_FinancialStatement ReportType", "33 Sector(name)"], how='left')

        temp = feats.groupby(["Result_FinancialStatement FiscalYear", "Result_FinancialStatement ReportType", "33 Sector(name)"])[AGG_FEATURE].rank(ascending=False).add_prefix('quaterly_rank_33sec_')
        feats = pd.concat([feats, temp], axis=1)
        feats.set_index('datetime', inplace=True)

        for feature in PREV_YEAR_FEATURE:
            feats[f'rate_{feature}'] = (feats[feature] - feats[f'prev_year_{feature}']) / feats[f'prev_year_{feature}']

        for feature in QUARTER_FEATURE:
            feats[f'rate_prev_quarter_{feature}'] = (feats[feature] - feats[f'quarter_{feature}']) / feats[f'quarter_{feature}']

        unused_feature = UNUSED_FEATURE + [f'prev_year_{feature}' for feature in PREV_YEAR_FEATURE]
        feature_cols = feats.columns[feats.columns.isin(unused_feature) == False]

        cat_cols = [col for col in feature_cols if feats[col].dtype in ['O']]

        for feature_name in cat_cols:
            feats.loc[feats[feature_name].isnull(), feature_name] = 'N/A'
            feats[feature_name] = feats[feature_name].astype(str)
            feats = label_encode(feats, feature_name, model_path)

        # ??????????????????csv?????????????????????
        # ?????????:datetime???code?????????????????????(Ex 2016-05-09-1301)
        # ?????????:label_high_20????????????????????????????????????
        # ?????????:label_low_20????????????????????????????????????
        # header????????????B???C??????float64

        # ???????????????????????????????????????
        df = feats.loc[:, ["code"]].copy()
        # code?????????????????????????????????????????????
        df.loc[:, "code"] = df.index.strftime("%Y-%m-%d-") + df.loc[:, "code"].astype(
            str
        )

        # ????????????????????????
        output_columns = ["code"]

        # ????????????????????????
        for label in labels:
            #m = os.path.join(model_path, f"qt_{label}.pkl")
            #with open(m, "rb") as f:
            #    qt = pickle.load(f)
            # ????????????
            df[label] = cls.models[label].predict(feats[feature_cols])
            #df[label] = qt.inverse_transform(y_pred.reshape(-1, 1))
            # ????????????????????????
            output_columns.append(label)

        out = io.StringIO()
        df.to_csv(out, header=False, index=False, columns=output_columns)

        return out.getvalue()