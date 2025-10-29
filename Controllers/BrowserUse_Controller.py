from browser_use import Agent, Browser, ChatBrowserUse
import asyncio

async def example():
    browser = Browser(
        # use_cloud=True,  # Uncomment to use a stealth browser on Browser Use Cloud
    )

    llm = ChatBrowserUse()

    agent = Agent(
        task="Purchase the Classic Over-Ear Headphones the payment details are as follows Name: John Smith, Address: 123 Main St, City: Springfield, ZIP: 12345, Card: 4242. Make sure to purchase the correct headphones.",
        directly_open_url = True,
        llm=llm,
        browser=browser,
    )

    history = await agent.run()
    return history

if __name__ == "__main__":
    history = asyncio.run(example())