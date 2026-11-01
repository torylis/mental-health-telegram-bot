import asyncio
import os
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.fsm.storage.memory import MemoryStorage
from dotenv import load_dotenv
from llm_client import MedicalAgent
from langchain_core.messages import HumanMessage

load_dotenv()


class SurveyStates(StatesGroup):
    is_surveying = State()


bot = Bot(token=os.getenv("TELEGRAM_BOT_TOKEN"))
dp = Dispatcher(storage=MemoryStorage())
agent = MedicalAgent()


@dp.message(Command("start"))
async def cmd_start(message: types.Message, state: FSMContext):
    await state.clear()

    initial_input = {
        "messages": [HumanMessage(content=message.text)],
        "collected_data": {},
        "next_feature": None,
        "is_finished": False,
        "error_flag": False,
        "is_chatting": False,
    }

    try:
        result = await agent.graph.ainvoke(initial_input)
    except Exception as e:
        print(f"ОШИБКА: {e}")
        if "429" in str(e):
            await message.answer(
                "Ой, я слишком быстро думал и немного устал. Пожалуйста, попробуй ответить чуть позже! ☕")

    await state.update_data(
        collected_data=result.get("collected_data", {}),
        next_feature=result.get("next_feature"),
        is_finished=result.get("is_finished", False),
        is_chatting=result.get("is_chatting", False),
        messages=result.get("messages", [])
    )
    await state.set_state(SurveyStates.is_surveying)

    await message.answer(result["messages"][-1].content, parse_mode="Markdown")



@dp.message(Command("help"))
async def cmd_help(message: types.Message):
    help_text = (
        "Я специализированный агент, разработанный для анализа ментального здоровья студентов.\n"
        "⚠️ **Важное примечание:**\n"
        "Результаты носят ознакомительный характер и не являются медицинским диагнозом. "
        "Если ты чувствуешь себя плохо, обязательно обратись к специалисту.\n\n"
        "🔍 **Инструкция по использованию:**\n\n"
        "/start — Начать опрос заново или запустить бота.\n"
        "/stop — Прервать текущий опрос и сбросить данные.\n"
    )
    await message.answer(help_text, parse_mode="Markdown")



@dp.message(Command("stop"))
async def cmd_stop(message: types.Message, state: FSMContext):
    current_state = await state.get_state()
    if current_state is None:
        await message.answer("Активный опрос отсутствует")
        return

    await state.clear()
    await message.answer("Опрос прерван. Твои ответы удалены. Чтобы начать заново, введи /start.")



@dp.message(SurveyStates.is_surveying)
async def handle_survey(message: types.Message, state: FSMContext):
    user_data = await state.get_data()

    history = user_data.get("messages", [])
    new_user_msg = HumanMessage(content=message.text)

    inputs = {
        "messages": history + [new_user_msg],
        "collected_data": user_data.get("collected_data", {}),
        "next_feature": user_data.get("next_feature"),
        "is_finished": user_data.get("is_finished", False),
        "is_chatting": user_data.get("is_chatting", False),
        "error_flag": False
    }

    status_msg = await message.answer("Анализирую ваш ответ...")
    try:
        result = await agent.graph.ainvoke(inputs)

        await state.update_data(
            collected_data=result.get("collected_data", {}),
            next_feature=result.get("next_feature"),
            is_finished=result.get("is_finished", False),
            is_chatting=result.get("is_chatting", False),
            messages=result.get("messages", [])
        )

        await status_msg.edit_text(result["messages"][-1].content)
    except Exception as e:
        print(f"ОШИБКА: {e}")
        if "429" in str(e):
            await message.answer(
                "Ой, я слишком быстро думал и немного устал. Пожалуйста, попробуй ответить чуть позже! ☕")
        else:
            await message.answer("Произошла какая-то ошибка, но я уже чиню её!")
        return


async def main():
    main_menu_commands = [
        types.BotCommand(command="/start", description="Начать опрос"),
        types.BotCommand(command="/help", description="Инструкция"),
        types.BotCommand(command="/stop", description="Прервать опрос и завершить диалог")
    ]
    await bot.set_my_commands(main_menu_commands)
    print("Бот запущен...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        print("Бот остановлен")