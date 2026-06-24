from Main_Binance import run_binance
# from Main_Interactive import run_interactive

if __name__ == "__main__":
    Broker = "Binance"
    # Broker = "Interactive Broker"

    if Broker == "Binance":
        run_binance(Broker=Broker)
    # elif Broker == "Interactive Broker":
    #     run_interactive()

