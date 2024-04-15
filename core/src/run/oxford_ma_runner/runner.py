from run.runner import (Runner, RunningSessionParams, SessionPartialResults,
                        SessionResults)


class OxfordMARunner(Runner):
    def __init__(self, params: RunningSessionParams) -> None:
        super(params)

    async def run(self) -> SessionResults:
        print("Running!")
        return SessionResults(models=[], train_metadata={})  # type:ignore

    async def stop(self) -> SessionPartialResults:
        print("Stopping!")
        return SessionPartialResults(train_metadata={})
