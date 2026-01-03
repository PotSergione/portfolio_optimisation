import asyncio
from telegram import Bot


    # MSG = f"{datetime.datetime.now()}\n" +\
    # '\n--- Optimal Hybrid Portfolio Found --- \n' +\
    # f"Expected Daily Return: {expected_return:.2%}\n" +\
    # f"Daily Volatility: {daily_vol:.2%}\n"+\
    # f"Daily CVaR ({int(confidence_level*100)}%): {cvar_alpha:.2%}\n"+\
    # f"\nAnnualized Return: {annual_return:.2%}\n"+\
    # f"Annualized Volatility: {annual_vol:.2%}\n"+\
    # f"Annualized CVaR ({int(confidence_level*100)}%): {annual_cvar:.2%}\n" +\
    # '\n---  Copula model fit of the data ---\n' +\
    # "Original Spearman Correlation:\n" +\
    # f"{df_daily.corr(method='spearman').iloc[0,1]}\n"+\
    # "Simulated Spearman Correlation:\n"+\
    # f"{df_sim.corr(method='spearman').iloc[0,1]}\n"+\
    # "\n--- Optimal picks ---\n"

    # for i, w in enumerate(opt_weights):
    #     if np.round(w, 2) > 0.00:
    #         MSG += f"{tickers[i]} Weight: {w:.2%}\n"

async def broadcast_message(token, chat_ids, message, retries=3):

    bot = Bot(token=token)
    # --- CONFIGURATION ---
    MY_TOKEN = ''
    # In a real app, you'd load these from a database or file
    SUBSCRIBERS = []
    for chat_id in chat_ids:
        for attempt in range(retries):
            try:
                await bot.send_message(chat_id=chat_id, text=message)
                print(f"Sent to {chat_id}")
                await asyncio.sleep(0.05)
                break  # Success, exit retry loop
            except Exception as e:
                if attempt < retries - 1:
                    print(f"Attempt {attempt + 1} failed for {chat_id}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
                else:
                    print(f"Failed to send to {chat_id} after {retries} attempts: {e}")

