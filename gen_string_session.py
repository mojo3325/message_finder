import os
from telethon import TelegramClient
from telethon.sessions import StringSession


def main() -> None:
    api_id = os.getenv("TELEGRAM_API_ID") or input("TELEGRAM_API_ID: ")
    api_hash = os.getenv("TELEGRAM_API_HASH") or input("TELEGRAM_API_HASH: ")
    with TelegramClient(StringSession(), int(api_id), api_hash) as client:
        print("Your StringSession:")
        print(client.session.save())


if __name__ == "__main__":
    main()


