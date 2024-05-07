import asyncio
import signal
from types import FrameType

from seg_tgce.run.oxford_ma_runner.runner import OxfordMARunner
from seg_tgce.run.runner import RunningSessionParams

runner = OxfordMARunner(
    params=RunningSessionParams(
        n_epochs=50,
        target_img_shape=(256, 256),
        batch_size=32,
        num_annotators=4,
        extra={
            "noise_levels_snr": [20, 10, 0, -10],
            "entropy_gamma_values": [1e-6, 1e-3, 1e-1],
        },
    )
)


async def interruption_handler(  # pylint: disable=unused-argument
    signum: int,
    fram: FrameType | None,
) -> None:
    partial_res = await runner.stop()
    print(partial_res)


async def main() -> None:
    signal.signal(signal.SIGINT, interruption_handler)
    results = await runner.run()
    print(results)


if __name__ == "__main__":
    asyncio.run(main())
