# file: list_human_privates.py
import asyncio
import os
import argparse
from telethon import TelegramClient
from telethon.tl.types import User

def parse_args():
    p = argparse.ArgumentParser(description="Вывести приватные чаты только с людьми (не ботами).")
    p.add_argument("--api-id", type=int, default=int(os.getenv("TG_API_ID", "0")), help="Telegram api_id")
    p.add_argument("--api-hash", default=os.getenv("TG_API_HASH", ""), help="Telegram api_hash")
    p.add_argument("--session", default=os.getenv("TG_SESSION", "human_only"), help="Имя файла сессии")
    p.add_argument("--strict", action="store_true",
                   help="Строгий режим: исключить deleted/fake/scam аккаунты")
    p.add_argument("--csv", default=None, help="Путь для выгрузки CSV")
    return p.parse_args()

def format_user(u: User) -> str:
    name = " ".join(filter(None, [u.first_name, u.last_name])) or "(без имени)"
    uname = f"@{u.username}" if u.username else ""
    phone = u.phone or ""
    return f"{u.id}\t{name}\t{uname}\t{phone}"

async def main():
    args = parse_args()
    if not args.api_id or not args.api_hash:
        raise SystemExit("Укажи --api-id и --api-hash или переменные окружения TG_API_ID/TG_API_HASH.")

    client = TelegramClient(args.session, args.api_id, args.api_hash)
    await client.start()  # первичный логин спросит код/пароль

    humans = []
    async for dialog in client.iter_dialogs():
        # Нас интересуют только приватные диалоги с пользователями
        # dialog.entity будет User для приваток
        ent = dialog.entity
        if not isinstance(ent, User):
            continue  # группы/каналы лесом
        if getattr(ent, "bot", False):
            continue  # боты не люди, сюрприз

        if args.strict:
            if getattr(ent, "deleted", False):
                continue
            if getattr(ent, "scam", False):
                continue
            if getattr(ent, "fake", False):
                continue

        humans.append(ent)

    # Вывод
    if not humans:
        print("Ничего человеческого не найдено. Зато честно.")
    else:
        print("id\tname\tusername\tphone")
        for u in humans:
            print(format_user(u))

    # CSV при необходимости
    if args.csv:
        import csv
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f, delimiter=",")
            w.writerow(["id", "name", "username", "phone"])
            for u in humans:
                name = " ".join(filter(None, [u.first_name, u.last_name])) or ""
                uname = u.username or ""
                phone = u.phone or ""
                w.writerow([u.id, name, uname, phone])
        print(f"CSV сохранен: {args.csv}")

if __name__ == "__main__":
    asyncio.run(main())
