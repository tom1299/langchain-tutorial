"""
Examples from https://reference.langchain.com/python/langchain-core/runnables/base/Runnable
"""
import pytest
from langchain_core.globals import set_debug
from langchain_core.runnables import RunnableLambda

set_debug(True)

def hello(name: str) -> str:
    return "Hello " + name

class TestRunnableExamples:

    def test_runnable_with_function(self):
        runnable = RunnableLambda(hello)
        result = runnable.invoke("Alice")
        assert result == "Hello Alice"

    def test_runnable_sequence(self):

        sequence = RunnableLambda(lambda x: x + 1) | RunnableLambda(lambda x: x * 2)
        sequence.invoke(1)  # 4
        sequence.batch([1, 2, 3])  # [4, 6, 8]

        sequence = RunnableLambda(lambda x: x + 1) | {
            "mul_2": RunnableLambda(lambda x: x * 2),
            "mul_5": RunnableLambda(lambda x: x * 5),
        }
        result = sequence.invoke(1)  # {'mul_2': 4, 'mul_5': 10}
        print(result)

        assert result == {'mul_2': 4, 'mul_5': 10}

    @pytest.mark.asyncio
    async def test_runnable_parallel(self):

        def add_one(x: int) -> int:
            return x + 1

        def mul_two(x: int) -> int:
            return x * 2

        def mul_three(x: int) -> int:
            return x * 3

        runnable_1 = RunnableLambda(add_one)
        runnable_2 = RunnableLambda(mul_two)
        runnable_3 = RunnableLambda(mul_three)

        sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
            "mul_two": runnable_2,
            "mul_three": runnable_3,
        }

        sequence.invoke(1)
        await sequence.ainvoke(1)

        sequence.batch([1, 2, 3])
        await sequence.abatch([1, 2, 3])
