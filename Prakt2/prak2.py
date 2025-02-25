import asyncio
import httpx
from aiogram import Bot, Dispatcher, types
from aiogram.types import ReplyKeyboardMarkup, KeyboardButton

#токен бота
TOKEN = "7624245583:AAFFC3unv8sAJAD2MNLWYZ7NlvQOHKEb_5U"

# Адрес ESP сервера (через WOKWIgw)
ESP_URL = "http://localhost:9080"

bot = Bot(token=TOKEN)
dp = Dispatcher()


main_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="Ввімкнути Вентилятор")],
        [KeyboardButton(text="Вимкнути Вентилятор")],
        [KeyboardButton(text="Тимчасово ввімкнути")]
    ],
    resize_keyboard=True
)

# Меню для таймеру
temp_keyboard = ReplyKeyboardMarkup(
    keyboard=[
        [KeyboardButton(text="10 хв")],
        [KeyboardButton(text="30 хв")],
        [KeyboardButton(text="1 год")],
        [KeyboardButton(text="Назад")]
    ],
    resize_keyboard=True
)

async def send_command(command: str) -> str:
   
    #Відправка GET-запиту ESP з командою і повертає відповідь

    url = f"{ESP_URL}/{command}"
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(url)
            return r.text
    except Exception as e:
        return f"Невдалося з`єднатися з платою: {str(e)}"

@dp.message()
async def handle_message(message: types.Message):
    if message.text == "Ввімкнути Вентилятор":
        response = await send_command("TURN_ON")
        await message.answer(
            f"Команду успішно відправлено.\nВідповідь плати: {response}",
            reply_markup=main_keyboard
        )

    elif message.text == "Вимкнути Вентилятор":
        response = await send_command("TURN_OFF")
        await message.answer(
            f"Команду успішно відправлено.\nВідповідь плати: {response}",
            reply_markup=main_keyboard
        )

    elif message.text == "Тимчасово ввімкнути":
        await message.answer("Оберіть час увімкнення:", reply_markup=temp_keyboard)

    elif message.text == "10 хв":
        response = await send_command("TIMER_10")
        await message.answer(
            f"Вентилятор увімкнено на 10 хв.\nВідповідь плати: {response}",
            reply_markup=main_keyboard
        )

    elif message.text == "30 хв":
        response = await send_command("TIMER_30")
        await message.answer(
            f"Вентилятор увімкнено на 30 хв.\nВідповідь плати: {response}",
            reply_markup=main_keyboard
        )

    elif message.text == "1 год":
        response = await send_command("TIMER_60")
        await message.answer(
            f"Вентилятор увімкнено на 1 год.\nВідповідь плати: {response}",
            reply_markup=main_keyboard
        )

    elif message.text == "Назад":
        await message.answer("Повернення в головне меню", reply_markup=main_keyboard)

    else:
        await message.answer("Оберіть одну з кнопок нижче:", reply_markup=main_keyboard)

async def main():
    await dp.start_polling(bot)

if __name__ == "__main__":
    asyncio.run(main())
