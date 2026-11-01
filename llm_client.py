import os
from typing import TypedDict, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import joblib
import pandas as pd
from langgraph.graph.message import add_messages
from typing import Annotated, Sequence
from langchain_core.messages import BaseMessage, SystemMessage

load_dotenv()


class AgentState(TypedDict):
    collected_data: Dict[str, Any]
    next_feature: Optional[str]
    is_finished: bool
    error_flag: bool
    is_chatting: bool
    messages: Annotated[Sequence[BaseMessage], add_messages]


FEATURES_LIST = [
    "Gender", "Age", "Academic Pressure", "Study Satisfaction",
    "Sleep Duration", "Dietary Habits", "Have you ever had suicidal thoughts ?",
    "Study Hours", "Financial Stress", "Family History of Mental Illness"
]


class MedicalAgent:
    FEATURE_NAMES_RU = {
        "Gender": "Пол",
        "Age": "Возраст",
        "Academic Pressure": "Учебная нагрузка",
        "Study Satisfaction": "Удовлетворенность учебой",
        "Sleep Duration": "Часы сна в день",
        "Dietary Habits": "Режим питания",
        "Have you ever had suicidal thoughts ?": "Наличие суицидальных мыслей",
        "Study Hours": "Часов учебы в день",
        "Financial Stress": "Финансовый стресс",
        "Family History of Mental Illness": "Ментальные заболевания в семье"
    }
    FEATURES_HINTS = {
        "Gender": "спроси пол пользователя (мужской или женский).",
        "Age": "уточни возраст пользователя.",
        "Academic Pressure": "попроси оценить уровень учебной нагрузки от 1 до 5 (где 5 — очень тяжело, 1 - очень легко).",
        "Study Satisfaction": "узнай, насколько нравится учеба, по шкале от 1 до 5 (где 5 - очень нравится, 1 — совсем не нравится).",
        "Sleep Duration": "спроси, сколько часов в день в среднем длится сон.",
        "Dietary Habits": "уточни, как пользователь питается (здоровое питание, вредная еда или среднее).",
        "Have you ever had suicidal thoughts ?": "осторожно и тактично спроси, бывали ли когда-нибудь мысли о причинении себе вреда.",
        "Study Hours": "узнай, сколько примерно часов в день уходит на учебу.",
        "Financial Stress": "попроси оценить уровень финансового стресса от 1 до 5.",
        "Family History of Mental Illness": "спроси, были ли у близких родственников проблемы с ментальным здоровьем."
    }
    FEATURE_EXTRACTION_RULES = {
        "Gender": "Верни 0 для мужского пола (Male, мужчина, парень, м), "
                  "Верни 1 для женского пола (Female, женщина, девушка, ж).",
        "Age": "Извлеки целое число лет. Например: 'мне 20' -> 20",
        "Academic Pressure": "Выбери наиболее подходящие целое число по шкале из ответа пользователя"
                             "Шкала 1-5. 1: почти нет нагрузки, 2-3: средняя, 4-5: очень тяжело/выгорание.",
        "Study Satisfaction": "Выбери наиболее подходящие целое число по шкале из ответа пользователя"
                              "Шкала 1-5. 1: ужасно/хочу отчислиться, 2-3: нормально, 4-5: очень нравится.",
        "Sleep Duration": "0 для количества сна <5ч, "
                          "1 для 5-6ч, "
                          "2 для 7-8ч, "
                          "3: для >8ч.",
        "Dietary Habits": "0 - плохое питание, фастфуд, "
                          "1 - обычная еда, нормальное питание, "
                          "2 - здорое питание, ПП, полезная еда",
        "Have you ever had suicidal thoughts?": "0 - нет, никогда не было, "
                                                "1 - да, были мысли, задумывался.",
        "Study Hours": "Извлеки количество часов учебы в день как целое число.",
        "Financial Stress": "Выбери наиболее подходящие целое число по шкале из ответа пользователя"
                            "Шкала 1-5. 1: нет проблем, 2-3: средний стресс, 4-5: очень тяжело/долги.",
        "Family History of Mental Illness": "0 - нет, в семье здоровы, проблем ни у кого не было, "
                                            "1 - да, были диагнозы/зависимости)."
    }

    def __init__(self):
        self.llm = ChatOpenAI(
            model="openai/gpt-oss-20b:free",
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY")
        )
        self.rf_model = joblib.load("rf_model.pkl")
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)

        workflow.add_node("extract", self.extract_node)
        workflow.add_node("router", self.determine_next_step_node)
        workflow.add_node("ask", self.ask_node)
        workflow.add_node("predict", self.predict_node)
        workflow.add_node("chat", self.chat_node)

        workflow.set_entry_point("extract")
        workflow.add_conditional_edges(
            "extract",
            lambda x: "chat" if x.get("is_chatting") else "router",
            {
                "chat": "chat",
                "router": "router"
            }
        )
        workflow.add_conditional_edges(
            "router",
            lambda x: "predict" if x.get("is_finished") else "ask",
            {
                "predict": "predict",
                "ask": "ask"
            }
        )
        workflow.add_edge("ask", END)
        workflow.add_edge("predict", END)
        workflow.add_edge("chat", END)

        return workflow.compile()



    def validate_value(self, feature: str, val: int) -> bool:
        if feature in ["Gender", "Have you ever had suicidal thoughts ?", "Family History of Mental Illness"]:
            return val in [0, 1]

        if feature in ["Academic Pressure", "Study Satisfaction", "Financial Stress"]:
            return 1 <= val <= 5

        if feature == "Sleep Duration":
            return 0 <= val <= 3

        if feature == "Dietary Habits":
            return 0 <= val <= 2

        if feature == "Age":
            return 0 < val < 120

        if feature in "Study Hours":
            return 0 <= val <= 24

        return False



    async def extract_node(self, state: AgentState):
        f = state.get("next_feature")
        if not f:
            return {"error_flag": False}

        f_ru = self.FEATURE_NAMES_RU.get(f, f)
        last_msg_obj = state["messages"][-1]
        msg = last_msg_obj.content

        specific_rule = self.FEATURE_EXTRACTION_RULES.get(f, "Извлеки числовое значение.")

        mapping_instruction = f"""
            Ты — высокоточный классификатор данных.
            Твоя цель — извлечь значение для параметра "{f_ru}" из сообщения: "{msg}".

            ПРАВИЛО КЛАССИФИКАЦИИ ДЛЯ "{f_ru}":
            {specific_rule}

            ИНСТРУКЦИЯ:
            1. Если в сообщении есть ответ — верни ТОЛЬКО ЧИСЛО.
            2. Если сообщение не содержит ответа или это бессмыслица — верни 'INVALID'.
            3. Не пиши никаких пояснений, только число или слово INVALID.
            """

        res = await self.llm.ainvoke(mapping_instruction)
        val = res.content.strip()

        if val == "INVALID":
            return {"error_flag": True}
        else:
            try:
                val_int = int(float(val))
                if self.validate_value(f, val_int):
                    new_data = state["collected_data"].copy()
                    new_data[f] = int(float(val_int))
                    return {"collected_data": new_data, "error_flag": False}
                else:
                    return {"error_flag": True}
            except:
                return {"error_flag": True}



    def determine_next_step_node(self, state: AgentState):
        for feature in FEATURES_LIST:
            if feature not in state["collected_data"]:
                return {"next_feature": feature, "is_finished": False}
        return {"is_finished": True, "next_feature": None}



    async def ask_node(self, state: AgentState):
        feature = state["next_feature"]
        feature_ru = self.FEATURE_NAMES_RU.get(feature, feature)
        error = state.get("error_flag", False)
        collected = state.get("collected_data", {})
        last_msg_obj = state["messages"][-1]
        last_msg = last_msg_obj.content

        hint = self.FEATURES_HINTS.get(feature, f"задай вопрос про {feature}")

        if error:
            prompt = f"""
            Пользователь дал некорректный ответ на вопрос о '{feature_ru}'.
            Его ответ был {last_msg}.
            Твоя задача: вежливо сказать, что ты не совсем понял ответ, 
            и попросить уточнить информацию именно по параметру '{feature_ru}'.
            Не предлагай не отвечать на вопрос.
            {hint}
            ПРАВИЛА ОФОРМЛЕНИЯ:
            1. НЕ используй символы '*' или '_' для выделения текста.
            2. Пиши текст просто и понятно, без Markdown-разметки.
            3. Пиши только на русском.
            """
        elif len(collected) == 0:
            prompt = f"""
            Ты начинаешь опрос студента для оценки риска депрессии.
            Поприветствуй его. 
            Расскажи что ты ИИ-ассистент, помогающий студентам оценить их ментальное состояние,
            и что ты проведешь небольшой опрос, чтобы узнать о его состоянии.
            Попроси ответить на вопрос, не предлагай на него не отвечать.
            {hint}
            ПРАВИЛА ОФОРМЛЕНИЯ:
            1. НЕ используй символы '*' или '_' для выделения текста.
            2. Пиши текст просто и понятно, без Markdown-разметки.
            3. Пиши только на русском.
            """
        else:
            prompt = f"""
            Ты продолжаешь опрос студента для оценки риска депрессии. 
            Задай следующий вопрос про '{feature_ru}'. 
            Не предлагай не отвечать на вопрос.
            {hint}
            ПРАВИЛА ОФОРМЛЕНИЯ:
            1. НЕ используй символы '*' или '_' для выделения текста.
            2. Пиши текст просто и понятно, без Markdown-разметки.
            3. Пиши только на русском.
            """

        res = await self.llm.ainvoke(prompt)
        return {"messages": [res], "error_flag": False}



    def run_prediction(self, collected_data):
        df = pd.DataFrame([collected_data])
        df = df[FEATURES_LIST]
        return self.rf_model.predict_proba(df)[0][1]



    async def predict_node(self, state: AgentState):
        data = state["collected_data"]
        probability = self.run_prediction(data)

        prompt = f"""
            Ты — поддерживающий ИИ-консультант. 
            Модель предсказала риск депрессии {probability * 100:.0f}%. 
            Напиши эмпатичный вывод и дай совет студенту, опираясь на информацию из переписки.
            В конце своего ответа обязательно поинтересуйся у пользователя, как у него дела с курсовой работой. 
            Скажи, что если она еще даже не начата — это нормально и переживать не стоит,
            и у некоторых студентов еще даже не известна тема курсовой. 
            Добавь дисклеймер, что это не медицинский диагноз.
            Пиши только на русском и не оформляй ответ в таблицы.
            ОГРАНИЧЕНИЕ: Не более 4000 символов.
            Добавь в самом конце, что ты готов пообщаться и ответить на любые вопросы.
            """

        full_history = [SystemMessage(content=prompt)] + state["messages"]

        res = await self.llm.ainvoke(full_history)
        return {"messages": [res], "is_chatting": True}



    async def chat_node(self, state: AgentState):
        system_instructions = f"""
        Ты — эмпатичный ИИ-помощник. Опрос уже завершен.
        Отвечай на любые вопросы студента, поддерживай его, давай советы по учебе и отдыху.
        Используй историю диалога.
        ПРАВИЛА ОФОРМЛЕНИЯ:
        1. НЕ используй символы '*' или '_' для выделения текста.
        2. Пиши текст просто и понятно, без Markdown-разметки.
        3. Пиши только на русском.
        """

        full_history = [SystemMessage(content=system_instructions)] + state["messages"]

        res = await self.llm.ainvoke(full_history)
        return {"messages": [res]}