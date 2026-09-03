# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Winner-only resolution of deterministic collection inputs."""

from __future__ import annotations

from data_designer.slurm.contracts import ArtifactReference
from data_designer.slurm.planning import ResolvedSlurmRunPlan
from data_designer.slurm.state.collection_validation import validate_collection_inputs
from data_designer.slurm.state.errors import StateCorruptionError, StateNotFoundError
from data_designer.slurm.state.finalization import WinnerFinalizer
from data_designer.slurm.state.outputs import CandidateOutputManifest, CollectionPlan, CollectionShard, ShardWinner
from data_designer.slurm.state.reader import StateReader
from data_designer.slurm.state.storage import StateStorage
from data_designer.slurm.state.validation import StateContractError, validate_collection_plan


class CollectionInputResolver:
    """Resolve planned winner chains without scanning attempt output trees."""

    def __init__(self, storage: StateStorage, reader: StateReader) -> None:
        self._storage = storage
        self._reader = reader
        self._finalizer = WinnerFinalizer(storage, reader)

    def get_winner_shards(self) -> tuple[CollectionShard, ...]:
        """Return one canonical winner reference for every ordered run shard."""
        run, plan, shards = self._reader.load_context()
        planned: list[CollectionShard] = []
        for shard in shards:
            attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            winner = self._finalizer.load_optional_winner(run, plan, shard, attempts)
            if winner is None:
                raise StateNotFoundError(f"shard {shard.shard_id!r} has no winner")
            planned.append(
                CollectionShard(
                    shard_id=shard.shard_id,
                    winner_manifest=ArtifactReference(
                        path=self._storage.get_winner_path(shard.shard_id).as_posix(),
                        sha256=winner.compute_sha256(),
                    ),
                )
            )
        return tuple(planned)

    def resolve(
        self, collection_plan: CollectionPlan
    ) -> tuple[ResolvedSlurmRunPlan, tuple[CandidateOutputManifest, ...]]:
        """Validate the complete winner chain and return ordered candidate manifests."""
        run, plan, shards = self._reader.load_context()
        winners: list[ShardWinner] = []
        candidates: list[CandidateOutputManifest] = []
        for planned_collection_shard, shard in zip(collection_plan.planned_shards, shards, strict=True):
            if planned_collection_shard.shard_id != shard.shard_id:
                raise StateCorruptionError("collection plan does not preserve planned shard order")
            attempts = self._reader.load_validated_shard_attempts(run, plan, shard)
            winner = self._finalizer.load_optional_winner(run, plan, shard, attempts)
            if winner is None:
                raise StateNotFoundError(f"shard {shard.shard_id!r} has no winner")
            expected_reference = ArtifactReference(
                path=self._storage.get_winner_path(shard.shard_id).as_posix(),
                sha256=winner.compute_sha256(),
            )
            if planned_collection_shard.winner_manifest != expected_reference:
                raise StateCorruptionError(f"collection winner changed for shard {shard.shard_id!r}")
            attempt = self._reader.get_attempt(attempts, winner.attempt_id)
            result = self._reader.load_optional_attempt_result(plan, shard, attempt)
            if result is None:
                raise StateCorruptionError(f"winning attempt {winner.attempt_id!r} has no result records")
            winners.append(winner)
            candidates.append(result[1])
        try:
            validate_collection_plan(run, collection_plan, shards, tuple(winners))
            validate_collection_inputs(plan, collection_plan, tuple(candidates))
        except StateContractError as error:
            raise StateCorruptionError("collection inputs violate persisted run intent") from error
        return plan, tuple(candidates)


__all__ = ["CollectionInputResolver"]
