import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

duration: int = 7  # minutes


def generate_cumulative_profit(
    filename: str = "data/USDCHF_M1_202501020000_202502131015.csv",
    duration: int = 7,
    output_image: bool = False
) -> float:
    df = pd.read_csv(filename, delimiter="\t")
    df.columns = ["date", "time", "open", "high", "low", "close", "tickvol", "vol", "spread"]

    df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
    # Set the datetime column as the index
    df.set_index('datetime', inplace=True)

    # Drop the original date and time columns
    df.drop(columns=['date', 'time'], inplace=True)

    df.drop(columns=['spread', 'tickvol', 'vol'], inplace=True)

    # Because we use `duration` min previous data to answer the current one.
    df[f'average_{duration}_min'] = df['close'].shift(1).rolling(window=duration).mean()

    df['signal'] = np.nan
    df['buy_signal'] = (df['open'] > df[f'average_{duration}_min']) & (
            df['open'].shift(1) < df[f'average_{duration}_min'])

    df['sell_signal'] = (df['open'] < df[f'average_{duration}_min']) & (
            df['open'].shift(1) > df[f'average_{duration}_min'])

    df['signal'] = 0
    df.loc[df['buy_signal'], 'signal'] = 1
    df.loc[df['sell_signal'], 'signal'] = -1

    # Calculate the returns based on the close price
    df['returns'] = df['close'].pct_change()

    # Calculate the profit or loss based on the signal
    df['profit_loss'] = df['returns'] * df['signal'].shift(1)

    # Calculate the cumulative profit or loss
    df['cumulative_profit_loss'] = (1 + df['profit_loss']).cumprod() - 1

    if output_image:
        plt.plot(df.index, df.cumulative_profit_loss)
        plt.savefig(f"duration_{duration}.png")
        plt.clf()

    return df['cumulative_profit_loss'].iloc[-1]


def optimize_duration(filename: str, min_duration: int, max_duration: int) -> int:
    best_duration = min_duration
    best_profit = float('-inf')

    for duration in range(min_duration, max_duration + 1):
        profit = generate_cumulative_profit(filename, duration)
        if profit > best_profit:
            best_profit = profit
            best_duration = duration

    return best_duration


# # USDCHF

# Example usage
filename = "data/USDCHF_M1_202501020000_202502131015.csv"
best_duration = optimize_duration(filename, 7, 90)
print(f"Best duration: {best_duration}")

# Best duration is 77 min.
generate_cumulative_profit(duration=77, output_image=True)

# # XAUUSD

filename = "data/XAUUSD_M1_202410310857_202502131055.csv"
best_duration = optimize_duration(filename, 7, 90)
print(f"Best duration: {best_duration}")

# Best duration is 34 min.
generate_cumulative_profit(duration=34, output_image=True)
