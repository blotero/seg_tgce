import asyncio
import signal

from run.oxford_ma_runner.runner import OxfordMARunner
from run.runner import RunningSessionParams

runner = OxfordMARunner(
    params=RunningSessionParams(
        n_epochs=50,
        target_img_shape=(64, 64),
        batch_size=32,
        num_annotators=4,
        extra={"annotators_noise_levels": [20, 10, 0, -10]},
    )
)


async def interruption_handler():
    partial_res = await runner.stop()
    print(partial_res)


async def main():
    signal.signal(signal.SIGINT, interruption_handler)
    results = await runner.run()
    print(results)


if __name__ == "__main__":

    asyncio.run(main())
