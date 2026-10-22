import datetime as dt
import heapq
import itertools
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Optional, Protocol

from tqdm import tqdm


class EventType(Enum):
    WAKEUP = auto()
    MESSAGE = auto()


AgentId = int

LatencyFn = Callable[[AgentId, AgentId], dt.timedelta]


@dataclass(order=True)
class Event:
    time: dt.datetime
    seq: int
    agent_id: AgentId = field(compare=False)
    etype: EventType = field(compare=False)
    payload: Any = field(compare=False, default=None)


class Oracle(Protocol):
    def initialize(self, start_time: dt.datetime) -> None:
        """Initialize the oracle at the start of the simulation."""
        ...

    def get_price(self, current_time: dt.datetime) -> float:
        """Return the reference price at current_time."""
        ...


class Agent(Protocol):
    def kernel_start(self, start_time: dt.datetime) -> None: ...
    def kernel_stop(self, stop_time: dt.datetime) -> None: ...
    def on_wakeup(self, current_time: dt.datetime) -> None: ...
    def on_message(self, current_time: dt.datetime, msg: Any) -> None: ...


class Kernel:
    def __init__(
        self,
        n_agents: int,
        *,
        latency_fn: Optional[LatencyFn] = None,
        default_latency: dt.timedelta = dt.timedelta(0),
        default_compute_delay: dt.timedelta = dt.timedelta(0),
    ):
        self.latency_fn: LatencyFn = latency_fn or (lambda s, r: default_latency)
        self.compute_delay: list[dt.timedelta] = [default_compute_delay] * n_agents

        self._queue: list[Event] = []
        self._seq = itertools.count()
        self._wakeup_seq: list[int] = [0] * n_agents
        self._extra_delay: dt.timedelta = dt.timedelta(0)

    def set_agents(self, agents: list[Agent]) -> None:
        self.agents = agents

    # agent interface

    def send_message(
        self,
        sender: AgentId,
        recipient: AgentId,
        msg: Any,
        delay: dt.timedelta = dt.timedelta(0),
    ) -> None:
        send_time = self.now + self.compute_delay[sender] + self._extra_delay + delay
        deliver_at = send_time + self.latency_fn(sender, recipient)
        self._push(deliver_at, recipient, EventType.MESSAGE, msg)

    def set_wakeup(self, sender: AgentId, at_time: dt.datetime) -> None:
        self._wakeup_seq[sender] += 1
        self._push(at_time, sender, EventType.WAKEUP, self._wakeup_seq[sender])

    def delay(self, extra: dt.timedelta) -> None:
        """temporary delay to separate simultaneous events"""
        self._extra_delay += extra

    def set_compute_delay(self, agent_id: AgentId, delay: dt.timedelta) -> None:
        self.compute_delay[agent_id] = delay

    # simulation

    def run(
        self,
        start_time: dt.datetime,
        stop_time: dt.datetime,
        progress_bar: bool = False,
    ) -> None:
        self.now = start_time
        self.stop_time = stop_time
        self._busy_until = [start_time] * len(self.agents)

        for agent in self.agents:
            agent.kernel_start(start_time)

        pbar = None
        if progress_bar:
            total_sec = (stop_time - start_time).total_seconds()
            pbar = tqdm(total=total_sec, desc="Simulating", unit="sim-s")

        last_time = start_time
        while self._queue and self._queue[0].time <= stop_time:
            event = heapq.heappop(self._queue)
            if (
                event.etype is EventType.WAKEUP
                and event.payload != self._wakeup_seq[event.agent_id]
            ):
                continue

            self.now = event.time
            agent_id = event.agent_id

            if pbar is not None and self.now > last_time:
                pbar.update((self.now - last_time).total_seconds())
                last_time = self.now

            if self._busy_until[agent_id] > self.now:
                self._push(
                    self._busy_until[agent_id], agent_id, event.etype, event.payload
                )
                continue

            self._extra_delay = dt.timedelta(0)
            agent = self.agents[agent_id]

            if event.etype is EventType.WAKEUP:
                agent.on_wakeup(self.now)
            else:
                agent.on_message(self.now, event.payload)

            self._busy_until[agent_id] = (
                self.now + self.compute_delay[agent_id] + self._extra_delay
            )

        if pbar is not None:
            pbar.update(max(0.0, pbar.total - pbar.n))
            pbar.close()

        for agent in self.agents:
            agent.kernel_stop(self.stop_time)

    # internal

    def _push(
        self,
        time: dt.datetime,
        agent_id: AgentId,
        etype: EventType,
        payload: Any = None,
    ) -> None:
        heapq.heappush(
            self._queue, Event(time, next(self._seq), agent_id, etype, payload)
        )
