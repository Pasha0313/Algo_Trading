import mplfinance as mpf
import pandas as pd
import matplotlib.pyplot as plt

def plot_live_data(df, target):
    #df_reset = df[target].copy().reset_index()  
    df_reset = df.copy().reset_index()  
    df_reset.loc[df_reset.index[-1], 'Date'] = df_reset.loc[df_reset.index[-1], 'Date'].value
    df_reset['Date'] = df_reset['Date'].astype('datetime64[ns]').astype('int64') // 10**9 
    df_reset['Date'] = pd.to_datetime(df_reset['Date'], unit='s')
    df = df_reset.set_index("Date")

    BACKGROUND_COLOUR="black"
    LINE_COLOUR = "blue"
    TEXT_COLOUR = "orange"

    style_dic = {"xtick.color": LINE_COLOUR, 
                "ytick.color": LINE_COLOUR, 
                "xtick.labelcolor": TEXT_COLOUR, 
                "ytick.labelcolor": TEXT_COLOUR,
                "axes.spines.top": False, 
                "axes.spines.right": False}

    market_colours = mpf.make_marketcolors(up="g", down="r", edge=BACKGROUND_COLOUR, wick=LINE_COLOUR)
    style = mpf.make_mpf_style(marketcolors=market_colours, facecolor=BACKGROUND_COLOUR, edgecolor=LINE_COLOUR,
                            figcolor=BACKGROUND_COLOUR, gridcolor=LINE_COLOUR, gridstyle="--", rc=style_dic)
    
    plt.ion()
    fig, ax = plt.subplots(figsize=(6, 3))
    # Clear the axes to prepare for new data
    ax.clear()

    # Plot the updated data on the same axes
    mpf.plot(df,type='candle',title="Live Candlestick Chart",ylabel="Price",style=style)

    # Redraw the plot
    plt.draw()
    plt.pause(0.5)


    
