"""
Пример топологии "звездочка" с медицинскими агентами.

Демонстрирует:
- Параллельное выполнение нескольких агентов (5 специалистов)
- Агрегацию результатов одним центральным агентом (терапевт)
- Сохранение истории диалога в JSON
- Топологию star: все специалисты → терапевт

Структура:
    Ортопед ─────┐
    Окулист ─────┤
    Кардиолог ───┼──→ Терапевт (финальный диагноз)
    Невролог ────┤
    Дерматолог ──┘
"""

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from rustworkx_framework.builder import GraphBuilder
from rustworkx_framework.core.graph import RoleGraph
from rustworkx_framework.execution import LLMCallerFactory, MACPRunner, RunnerConfig

# Глобальное хранилище для логирования промптов и ответов
AGENT_IO_LOG = {}


def create_logging_caller_factory():
    """Создать LLM caller factory с логированием входа/выхода."""
    # Создаем базовую фабрику
    base_factory = LLMCallerFactory.create_openai_factory()

    class LoggingFactory:
        """Обертка фабрики, которая логирует промпты и ответы."""

        def __init__(self, base_factory):
            self.base_factory = base_factory
            # Пробрасываем все атрибуты базовой фабрики
            self.default_caller = base_factory.default_caller
            self.default_async_caller = base_factory.default_async_caller
            self.default_config = base_factory.default_config
            self.caller_builder = base_factory.caller_builder
            self.async_caller_builder = base_factory.async_caller_builder

        def get_caller(self, config=None, agent_id=None):
            """Получить caller с логированием."""
            base_caller = self.base_factory.get_caller(config, agent_id)

            if base_caller is None:
                return None

            def logging_caller(prompt: str) -> str:
                """Обертка caller, которая логирует промпт и ответ."""
                # Логируем входной промпт
                if agent_id:
                    if agent_id not in AGENT_IO_LOG:
                        AGENT_IO_LOG[agent_id] = {}
                    AGENT_IO_LOG[agent_id]["input_prompt"] = prompt
                    AGENT_IO_LOG[agent_id]["input_length"] = len(prompt)

                # Вызываем реальный LLM
                response = base_caller(prompt)

                # Логируем ответ
                if agent_id:
                    AGENT_IO_LOG[agent_id]["output_response"] = response
                    AGENT_IO_LOG[agent_id]["output_length"] = len(response)

                return response

            return logging_caller

        def get_async_caller(self, config=None, agent_id=None):
            """Получить async caller с логированием."""
            return self.base_factory.get_async_caller(config, agent_id)

        def get_streaming_caller(self, config=None, agent_id=None):
            """Получить streaming caller с логированием."""
            return self.base_factory.get_streaming_caller(config, agent_id)

        def get_async_streaming_caller(self, config=None, agent_id=None):
            """Получить async streaming caller с логированием."""
            return self.base_factory.get_async_streaming_caller(config, agent_id)

    return LoggingFactory(base_factory)


# Конфигурация LLM (можно использовать разные модели для разных агентов)
DEFAULT_LLM_CONFIG = {
    "api_key": os.getenv("LLM_API_KEY", ""),
    "base_url": os.getenv("LLM_BASE_URL", "https://api.openai.com/v1"),
    "model_name": os.getenv("LLM_MODEL", "gpt-4"),
}

# Случай пациента для диагностики
PATIENT_CASE = """
Пациент: Мужчина, 45 лет

Жалобы:
- Боль в правом колене при ходьбе (3 недели)
- Периодические головные боли
- Ухудшение зрения (размытость на дальних расстояниях)
- Повышенное давление (150/95)
- Сухость кожи на руках и локтях
- Общая усталость

Анамнез:
- Работа в офисе (сидячая работа)
- Занимается бегом (3 раза в неделю)
- Стресс на работе последние 2 месяца
- Наследственность: у отца гипертония
"""


def create_medical_graph() -> RoleGraph:
    """Создать граф с медицинскими агентами в топологии звездочка."""
    builder = GraphBuilder()

    # 1. ОРТОПЕД - специализация на опорно-двигательном аппарате
    builder.add_agent(
        "orthopedist",
        display_name="Ортопед",
        persona="Вы опытный врач-ортопед с 15-летним стажем",
        description=(
            "Проанализируйте состояние опорно-двигательного аппарата пациента. "
            "Обратите внимание на боли в суставах, мышцах, последствия физической активности. "
            "Предоставьте краткое заключение (3-5 предложений) с вашими наблюдениями и рекомендациями."
        ),
        llm_backbone=DEFAULT_LLM_CONFIG["model_name"],
        base_url=DEFAULT_LLM_CONFIG["base_url"],
        api_key=DEFAULT_LLM_CONFIG["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 2. ОКУЛИСТ (офтальмолог) - специализация на зрении
    builder.add_agent(
        "ophthalmologist",
        display_name="Офтальмолог",
        persona="Вы квалифицированный врач-офтальмолог",
        description=(
            "Оцените состояние зрения пациента и возможные проблемы с глазами. "
            "Проанализируйте симптомы, связанные со зрением. "
            "Предоставьте краткое заключение (3-5 предложений) с вашими наблюдениями и рекомендациями."
        ),
        llm_backbone=DEFAULT_LLM_CONFIG["model_name"],
        base_url=DEFAULT_LLM_CONFIG["base_url"],
        api_key=DEFAULT_LLM_CONFIG["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 3. КАРДИОЛОГ - специализация на сердечно-сосудистой системе
    builder.add_agent(
        "cardiologist",
        display_name="Кардиолог",
        persona="Вы специалист по сердечно-сосудистым заболеваниям",
        description=(
            "Проанализируйте состояние сердечно-сосудистой системы пациента. "
            "Обратите особое внимание на давление, наследственность, факторы риска. "
            "Предоставьте краткое заключение (3-5 предложений) с вашими наблюдениями и рекомендациями."
        ),
        llm_backbone=DEFAULT_LLM_CONFIG["model_name"],
        base_url=DEFAULT_LLM_CONFIG["base_url"],
        api_key=DEFAULT_LLM_CONFIG["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 4. НЕВРОЛОГ - специализация на нервной системе
    builder.add_agent(
        "neurologist",
        display_name="Невролог",
        persona="Вы врач-невролог, специалист по заболеваниям нервной системы",
        description=(
            "Оцените неврологические симптомы пациента. "
            "Проанализируйте головные боли, связь со стрессом, общее состояние нервной системы. "
            "Предоставьте краткое заключение (3-5 предложений) с вашими наблюдениями и рекомендациями."
        ),
        llm_backbone=DEFAULT_LLM_CONFIG["model_name"],
        base_url=DEFAULT_LLM_CONFIG["base_url"],
        api_key=DEFAULT_LLM_CONFIG["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 5. ДЕРМАТОЛОГ - специализация на коже
    builder.add_agent(
        "dermatologist",
        display_name="Дерматолог",
        persona="Вы врач-дерматолог с экспертизой в дерматологии",
        description=(
            "Проанализируйте состояние кожи пациента. "
            "Обратите внимание на симптомы сухости, возможные причины и связь с общим состоянием здоровья. "
            "Предоставьте краткое заключение (3-5 предложений) с вашими наблюдениями и рекомендациями."
        ),
        llm_backbone=DEFAULT_LLM_CONFIG["model_name"],
        base_url=DEFAULT_LLM_CONFIG["base_url"],
        api_key=DEFAULT_LLM_CONFIG["api_key"],
        temperature=0.3,
        max_tokens=500,
    )

    # 6. ТЕРАПЕВТ - центральный агент, агрегирующий все заключения
    builder.add_agent(
        "general_practitioner",
        display_name="Врач-терапевт",
        persona="Вы опытный врач-терапевт, координирующий работу специалистов",
        description=(
            "Вы получили заключения от всех специалистов. "
            "Ваша задача:\n"
            "1. Проанализировать все заключения специалистов\n"
            "2. Выявить связи между различными симптомами\n"
            "3. Поставить общий диагноз или предварительный диагноз\n"
            "4. Дать комплексные рекомендации по лечению и дальнейшим действиям\n\n"
            "Структурируйте ваш ответ:\n"
            "- АНАЛИЗ: краткий анализ заключений специалистов\n"
            "- ДИАГНОЗ: основной диагноз или предположения\n"
            "- РЕКОМЕНДАЦИИ: конкретные рекомендации пациенту\n"
        ),
        llm_backbone=DEFAULT_LLM_CONFIG["model_name"],
        base_url=DEFAULT_LLM_CONFIG["base_url"],
        api_key=DEFAULT_LLM_CONFIG["api_key"],
        temperature=0.2,  # Более низкая температура для итогового диагноза
        max_tokens=1500,
    )

    # Создаём топологию "звездочка": все специалисты → терапевт
    specialists = ["orthopedist", "ophthalmologist", "cardiologist", "neurologist", "dermatologist"]

    for specialist in specialists:
        builder.add_workflow_edge(specialist, "general_practitioner")

    # Добавляем задачу (случай пациента)
    builder.add_task(query=PATIENT_CASE, answer="")

    # Подключаем задачу ко всем специалистам (они все начинают с анализа случая)
    builder.connect_task_to_agents(agent_ids=specialists)

    return builder.build()


def save_dialogue_history(result, output_path: str = "medical_dialogue_history.json"):
    """Сохранить историю диалога в JSON файл."""
    timestamp = datetime.now().isoformat()

    # Формируем структурированную историю
    dialogue_history: dict = {
        "metadata": {
            "timestamp": timestamp,
            "topology": "star",
            "total_agents": len(result.execution_order),
            "execution_time_seconds": result.total_time,
            "total_tokens": result.total_tokens,
        },
        "patient_case": PATIENT_CASE,
        "specialists_consultation": {},
        "final_diagnosis": {
            "agent": result.final_agent_id,
            "output": {
                "response": result.final_answer,
                "length": len(result.final_answer),
            },
        },
        "execution_flow": {
            "execution_order": result.execution_order,
            "parallel_groups": [],  # Будет заполнено ниже
        },
        "metrics": {
            "total_time": result.total_time,
            "total_tokens": result.total_tokens,
            "topology_changed_count": result.topology_changed_count,
            "fallback_count": result.fallback_count,
        },
    }

    # Добавляем сообщения специалистов
    specialists = ["orthopedist", "ophthalmologist", "cardiologist", "neurologist", "dermatologist"]

    for specialist_id in specialists:
        if specialist_id in result.messages:
            # Определяем русское название специалиста
            display_names = {
                "orthopedist": "Ортопед",
                "ophthalmologist": "Офтальмолог",
                "cardiologist": "Кардиолог",
                "neurologist": "Невролог",
                "dermatologist": "Дерматолог",
            }

            specialist_data = {
                "display_name": display_names.get(specialist_id, specialist_id),
                "output": {
                    "response": result.messages[specialist_id],
                    "length": len(result.messages[specialist_id]),
                },
                "execution_index": result.execution_order.index(specialist_id)
                if specialist_id in result.execution_order
                else -1,
            }

            # Добавляем входные данные, если они были залогированы
            if specialist_id in AGENT_IO_LOG:
                io_data = AGENT_IO_LOG[specialist_id]
                specialist_data["input"] = {
                    "prompt": io_data.get("input_prompt", ""),
                    "length": io_data.get("input_length", 0),
                }

            dialogue_history["specialists_consultation"][specialist_id] = specialist_data

    # Добавляем входные данные для терапевта (финального агента)
    if result.final_agent_id in AGENT_IO_LOG:
        io_data = AGENT_IO_LOG[result.final_agent_id]
        dialogue_history["final_diagnosis"]["input"] = {
            "prompt": io_data.get("input_prompt", ""),
            "length": io_data.get("input_length", 0),
        }

    # Если есть информация о параллельных группах, добавляем её
    if result.step_results:
        parallel_info = []
        for agent_id, step_result in result.step_results.items():
            parallel_info.append(
                {
                    "agent": agent_id,
                    "start_time": getattr(step_result, "start_time", None),
                    "end_time": getattr(step_result, "end_time", None),
                }
            )
        dialogue_history["execution_flow"]["parallel_execution_info"] = parallel_info

    # Добавляем информацию о состоянии агентов (если доступно)
    if result.agent_states:
        dialogue_history["agent_states"] = result.agent_states

    # Сохраняем в файл
    output_file = Path(__file__).parent / output_path
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(dialogue_history, f, ensure_ascii=False, indent=2)

    return output_file


def print_formatted_results(result):
    """Красиво выводим результаты на экран."""
    # Порядок выполнения

    # Заключения специалистов

    specialists_info = [
        ("orthopedist", "🦴 ОРТОПЕД"),
        ("ophthalmologist", "👁️ ОФТАЛЬМОЛОГ"),
        ("cardiologist", "❤️ КАРДИОЛОГ"),
        ("neurologist", "🧠 НЕВРОЛОГ"),
        ("dermatologist", "🧴 ДЕРМАТОЛОГ"),
    ]

    for specialist_id, _specialist_name in specialists_info:
        if specialist_id in result.messages:
            pass

    # Финальный диагноз терапевта

    # Метрики

    if result.pruned_agents:
        pass


def main():
    """Запустить пример с медицинскими агентами в топологии звездочка."""
    # Показываем случай пациента

    # Шаг 1: Создание графа
    start_time = time.time()

    graph = create_medical_graph()

    time.time() - start_time

    # Показываем топологию
    specialists = ["orthopedist", "ophthalmologist", "cardiologist", "neurologist", "dermatologist"]
    for _specialist in specialists:
        pass

    # Шаг 2: Создание runner с поддержкой параллелизма

    # Очищаем лог перед запуском
    AGENT_IO_LOG.clear()

    # Создаем конфигурацию runner'а с поддержкой параллельного выполнения
    runner_config = RunnerConfig(
        timeout=120.0,
        adaptive=True,
        enable_parallel=True,  # Включаем параллельное выполнение
        max_parallel_size=5,  # Все 5 специалистов могут работать параллельно
        broadcast_task_to_all=True,  # Task query передаётся всем агентам
    )

    # Создаем factory с логированием входа/выхода
    factory = create_logging_caller_factory()

    runner = MACPRunner(
        llm_factory=factory,
        config=runner_config,
    )

    # Шаг 3: Запуск консультации

    execution_start = time.time()

    try:
        # Запускаем с финальным агентом = терапевт
        result = runner.run_round(graph, final_agent_id="general_practitioner")

        time.time() - execution_start

        # Выводим результаты
        print_formatted_results(result)

        # Сохраняем историю диалога в JSON
        save_dialogue_history(result)

        # Показываем фрагмент того, что сохранено для терапевта
        if "general_practitioner" in AGENT_IO_LOG:
            gp_input = AGENT_IO_LOG["general_practitioner"].get("input_prompt", "")
            if "Messages from other agents:" in gp_input:
                idx = gp_input.find("Messages from other agents:")
                gp_input[idx : idx + 400]
                if len(gp_input[idx:]) > 400:
                    pass
            else:
                pass

        # Итоговая статистика

        # Проверяем, что параллельное выполнение работало
        if len(specialists) > 1:
            specialists_in_order = [s for s in result.execution_order if s in specialists]
            if len(specialists_in_order) == len(specialists):
                pass

    except Exception:
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
