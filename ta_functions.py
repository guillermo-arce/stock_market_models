# -*- coding: utf-8 -*-

# Importing Libraries
import pandas as pd

# Technical Analysis library

from ta.trend import (
    MACD,
    EMAIndicator,
    SMAIndicator
)

from ta.volatility import (
    BollingerBands,
    AverageTrueRange
)

from ta.momentum import (
    ROCIndicator,
    RSIIndicator,
    StochRSIIndicator,
    WilliamsRIndicator
)

from ta.volume import (
    AccDistIndexIndicator,
    VolumePriceTrendIndicator
)

#%% Constants definition, just for usability  
TIME = "time"
OPEN = "open"
HIGH = "high"
LOW = "low"
CLOSE = "close"
VOLUME = "volume"

#%% Technical Indicators functions (7 ta indicators -> Used in Pytorch's model)
def add_reduced_ta(df):
    df = add_reduced_trend_ta(df, HIGH, LOW, CLOSE)
    df = add_reduced_volatility_ta(df, HIGH, LOW, CLOSE)
    df = add_reduced_momentum_ta(df, HIGH, LOW, CLOSE, VOLUME)
    df = add_reduced_volume_ta(df, HIGH, LOW, CLOSE, VOLUME)  
    return df

def add_reduced_trend_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # SMAs
    df[f"close_sma"] = SMAIndicator(
        close=df[close], window=4, fillna=fillna
    ).sma_indicator()
    df[f"{colprefix}trend_sma_slow"] = SMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).sma_indicator()

    # EMAs
    df[f"{colprefix}trend_ema_fast"] = EMAIndicator(
        close=df[close], window=12, fillna=fillna
    ).ema_indicator()
    df[f"{colprefix}trend_ema_slow"] = EMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).ema_indicator()

    return df

def add_reduced_volatility_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # Average True Range (ATR)
    df[f"{colprefix}volatility_atr"] = AverageTrueRange(
        close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
    ).average_true_range()
    
    return df

def add_reduced_momentum_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # Relative Strength Index (RSI)
    df[f"{colprefix}momentum_rsi"] = RSIIndicator(
        close=df[close], window=14, fillna=fillna
    ).rsi()
    
    return df

def add_reduced_volume_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # Volume Price Trend (VPT)
    df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).volume_price_trend()
    
    return df

#%% Technical Indicators functions (23 ta indicators -> Used in Tensorflow's model)

def add_all_ta(df):
    df = add_trend_ta(df, HIGH, LOW, CLOSE)
    df = add_volatility_ta(df, HIGH, LOW, CLOSE)
    df = add_momentum_ta(df, HIGH, LOW, CLOSE, VOLUME)
    df = add_volume_ta(df, HIGH, LOW, CLOSE, VOLUME)  
    return df

def add_trend_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # SMAs
    df[f"close_sma"] = SMAIndicator(
        close=df[close], window=4, fillna=fillna
    ).sma_indicator()
    df[f"{colprefix}trend_sma_slow"] = SMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).sma_indicator()
    
    # MACD
    indicator_macd = MACD(
        close=df[close], window_slow=26, window_fast=12, window_sign=9, fillna=fillna
    )
    df[f"{colprefix}trend_macd"] = indicator_macd.macd()
    df[f"{colprefix}trend_macd_signal"] = indicator_macd.macd_signal()
    df[f"{colprefix}trend_macd_diff"] = indicator_macd.macd_diff()

    # EMAs
    df[f"{colprefix}trend_ema_fast"] = EMAIndicator(
        close=df[close], window=12, fillna=fillna
    ).ema_indicator()
    df[f"{colprefix}trend_ema_slow"] = EMAIndicator(
        close=df[close], window=26, fillna=fillna
    ).ema_indicator()

    return df

def add_volatility_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # Average True Range (ATR)
    df[f"{colprefix}volatility_atr"] = AverageTrueRange(
        close=df[close], high=df[high], low=df[low], window=10, fillna=fillna
    ).average_true_range()
    
    # Bollinger Bands
    indicator_bb = BollingerBands(
        close=df[close], window=20, window_dev=2, fillna=fillna
    )
    df[f"{colprefix}volatility_bbm"] = indicator_bb.bollinger_mavg()
    df[f"{colprefix}volatility_bbh"] = indicator_bb.bollinger_hband()
    df[f"{colprefix}volatility_bbl"] = indicator_bb.bollinger_lband()
    df[f"{colprefix}volatility_bbw"] = indicator_bb.bollinger_wband()
    df[f"{colprefix}volatility_bbp"] = indicator_bb.bollinger_pband()
    df[f"{colprefix}volatility_bbhi"] = indicator_bb.bollinger_hband_indicator()
    df[f"{colprefix}volatility_bbli"] = indicator_bb.bollinger_lband_indicator()
    
    return df

def add_momentum_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # Relative Strength Index (RSI)
    df[f"{colprefix}momentum_rsi"] = RSIIndicator(
        close=df[close], window=14, fillna=fillna
    ).rsi()

    # Stoch RSI (StochRSI) -> STCK, STCD
    indicator_srsi = StochRSIIndicator(
        close=df[close], window=14, smooth1=3, smooth2=3, fillna=fillna
    )
    df[f"{colprefix}momentum_stoch_rsi"] = indicator_srsi.stochrsi()
    df[f"{colprefix}momentum_stoch_rsi_k"] = indicator_srsi.stochrsi_k()
    df[f"{colprefix}momentum_stoch_rsi_d"] = indicator_srsi.stochrsi_d()

    # Williams R Indicator (WR)
    df[f"{colprefix}momentum_wr"] = WilliamsRIndicator(
        high=df[high], low=df[low], close=df[close], lbp=14, fillna=fillna
    ).williams_r()
    
    # Rate Of Change (ROC)
    df[f"{colprefix}momentum_roc"] = ROCIndicator(
        close=df[close], window=12, fillna=fillna
    ).roc()
    
    return df

def add_volume_ta(
    df: pd.DataFrame,
    high: str,
    low: str,
    close: str,
    volume: str,
    fillna: bool = False,
    colprefix: str = "",
) -> pd.DataFrame:
    
    # Accumulation Distribution Index (AD)
    df[f"{colprefix}volume_adi"] = AccDistIndexIndicator(
        high=df[high], low=df[low], close=df[close], volume=df[volume], fillna=fillna
    ).acc_dist_index()

    # Volume Price Trend (VPT)
    df[f"{colprefix}volume_vpt"] = VolumePriceTrendIndicator(
        close=df[close], volume=df[volume], fillna=fillna
    ).volume_price_trend()
    
    return df
    
    