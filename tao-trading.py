#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "rich>=13.9.4",
#   "pyyaml>=6.0.2",
#   "bittensor>=9.0.0",
#   "asyncio>=3.4.3",
# ]
# [tool.uv]
# prerelease = "allow"
# ///

from dataclasses import dataclass
import asyncio
import logging
import math
from typing import List, Dict, Tuple, Any, TypedDict

import yaml
import bittensor as bt
from bittensor.core.async_subtensor import get_async_subtensor
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
import argparse


class Config(TypedDict):
    wallet: str
    amount_staked: float
    amount_unstaked: float
    ranking_beta: float
    drive: float
    rebalance_threshold: float
    validators: Dict[int, List[str]]
    ranks: List[int]  # Add ranks to Config


@dataclass
class SubnetStats:
    price: float
    emission: float
    raw_yield: float
    weight: float
    boost: float
    score: float
    name: str
    volume: float

    def compute_raw_yield(self, emission: float, price: float) -> float:
        """Safe computation of raw yield with zero division protection"""
        return 0.0 if emission <= 0 else (emission - price) / emission

    def compute_score(self, emission: float, price: float, boost: float) -> float:
        """Safe computation of score with zero division protection"""
        if emission <= 0 or (1 + boost) <= 0:
            return 0.0
        return (emission * (1 + boost) - price) / (emission * (1 + boost))


@dataclass
class ValidatorStats:
    address: str
    stake: float
    rewards: float
    uptime: float
    performance_score: float


VOLUME_ALPHA: float = 0.1

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("staking.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
console = Console()


def read_config(config_path: str) -> Config:
    """Read and validate configuration from config.yaml."""
    try:
        with open(config_path, "r") as f:
            config: Config = yaml.safe_load(f)
        required_keys = {
            "wallet",
            "amount_staked",
            "amount_unstaked",
            "ranking_beta",
            "drive",
            "ranks",  # Add ranks to required keys
            "validators",
        }
        if missing := required_keys - set(config.keys()):
            raise ValueError(f"Missing required config keys: {missing}")
        # Set a default for rebalance_threshold if not provided.
        if "rebalance_threshold" not in config:
            config["rebalance_threshold"] = 0.01
        return config
    except Exception as e:
        logger.critical(f"Failed to read config.yaml: {e}")
        raise


def compute_weights_from_ranks(ranks: List[int], beta: float) -> Dict[int, float]:
    """Compute exponential weights for subnet ranks."""
    N = len(ranks)
    scores = [N - idx for idx in range(N)]
    exp_scores = [math.exp(beta * s) for s in scores]
    total_exp = sum(exp_scores)
    normalized_weights = [s / total_exp for s in exp_scores]
    return {netuid: normalized_weights[i] for i, netuid in enumerate(ranks)}


class StakingOptimizer:
    """
    Optimizes staking allocations across subnets by staking TAO to the best subnet
    and unstaking from the worst performing subnet.
    """

    def __init__(self, config_path: str, cli_ranks_file: str | None = None):
        self.config: Config = read_config(config_path)
        self.wallet_name: str = self.config["wallet"]
        self.amount_staked: float = self.config["amount_staked"]
        self.amount_unstaked: float = self.config["amount_unstaked"]
        self.validators_mapping: Dict[int, List[str]] = self.config["validators"]
        self.ranking_beta: float = self.config["ranking_beta"]
        self.drive: float = self.config["drive"]
        self.rebalance_threshold: float = self.config.get("rebalance_threshold", 0.01)
        self.allowed_subnets: List[int] = self.config[
            "ranks"
        ]  # Use ranks directly from config
        self.weight_dict: Dict[int, float] = compute_weights_from_ranks(
            self.allowed_subnets, self.ranking_beta
        )
        self.total_allocated: float = 0.0
        self.sub: Any = None
        self.wallet: Any = None
        self.last_volume: Dict[int, float] = {}
        self.avg_vol_delta: Dict[int, float] = {}

    def print_table_rich(
        self,
        stake_info: Dict,
        stats: Dict[int, SubnetStats],
        rank_dict: Dict[int, int],
        balance: float,
        best_validators: (
            Dict[int, Tuple[str, float]] | None
        ) = None,  # Add this parameter
    ) -> None:
        """Print a Rich table showing staking allocations."""
        total_stake_value = 0.0
        total_stake = 0.0

        table = Table(
            title="Staking Allocations",
            header_style="bold white on dark_blue",
            box=box.SIMPLE_HEAVY,
        )
        table.add_column("Subnet", justify="right", style="bright_cyan")
        table.add_column("Name", justify="left", style="white")
        table.add_column(
            "Validator", justify="left", style="bright_yellow"
        )  # New column
        table.add_column("Boost", justify="right", style="yellow")
        table.add_column("Yield", justify="right", style="cyan")
        table.add_column("Score", justify="right", style="bright_magenta")
        table.add_column("Vol Delta", justify="right", style="bright_red")
        table.add_column("Emission", justify="right", style="red")
        table.add_column("Price", justify="right", style="green")
        table.add_column("Stake", justify="right", style="magenta")
        table.add_column("Stake Value", justify="right", style="bright_green")
        table.add_column("Rank", justify="right", style="bright_blue")

        for netuid in self.allowed_subnets:
            stake_obj = stake_info.get(netuid)
            stake_amt = float(stake_obj.stake) if stake_obj is not None else 0.0
            total_stake += stake_amt

            if netuid in stats:
                stat = stats[netuid]
                price = stat.price
                raw_yield = stat.raw_yield
                boost = stat.boost
                score = stat.score
                emission = stat.emission
                name = stat.name
                current_volume = stat.volume
            else:
                price = raw_yield = boost = score = emission = 0.0
                name = ""
                current_volume = 0.0

            last_vol = self.last_volume.get(netuid, current_volume)
            raw_delta = current_volume - last_vol
            prev_avg = self.avg_vol_delta.get(netuid, raw_delta)
            avg_delta = VOLUME_ALPHA * raw_delta + (1 - VOLUME_ALPHA) * prev_avg
            self.avg_vol_delta[netuid] = avg_delta
            self.last_volume[netuid] = current_volume

            rank = int(rank_dict.get(netuid, 0))
            stake_value = stake_amt * price
            total_stake_value += stake_value

            # Get validator info if available
            validator_info = (
                best_validators.get(netuid, ("None", 0.0))
                if best_validators
                else ("None", 0.0)
            )
            validator_addr = validator_info[0]
            # Truncate validator address for display
            short_addr = (
                f"{validator_addr[:6]}...{validator_addr[-4:]}"
                if validator_addr != "None"
                else "None"
            )

            table.add_row(
                str(netuid),
                name,
                short_addr,  # Add validator address
                f"{boost:.4f}",
                f"{raw_yield:.4f}",
                f"{score:.4f}",
                f"{avg_delta:.4f}",
                f"{emission:.4f}",
                f"{price:.4f}",
                f"{stake_amt:.4f}",
                f"{stake_value:.4f}",
                str(rank),
            )

        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            "",
            "",
            "",
            "",
            "",
            "",
            f"[bold]{total_stake:.4f}[/bold]",
            f"[bold]{total_stake_value:.4f}[/bold]",
            "",
        )

        summary = (
            f"[bold cyan]Wallet Balance:[/bold cyan] {balance:.4f} TAO    "
            f"[bold cyan]Total Stake Value:[/bold cyan] {total_stake_value:.4f} TAO"
        )
        console.print(Panel(summary, style="bold white"))
        console.print(table)

    async def get_subnet_stats(self) -> Tuple[Dict[int, SubnetStats], Dict[int, int]]:
        """Fetch and compute subnet statistics."""
        all_subnets = await self.sub.all_subnets()
        sorted_subnets = sorted(all_subnets, key=lambda s: float(s.price), reverse=True)
        rank_dict = {s.netuid: idx + 1 for idx, s in enumerate(sorted_subnets)}
        stats: Dict[int, SubnetStats] = {}
        for subnet in all_subnets:
            netuid = subnet.netuid
            if netuid == 0 or netuid not in self.allowed_subnets:
                continue
            price = float(subnet.price)
            if price <= 0:
                continue
            emission = float(subnet.tao_in_emission)
            raw_yield = (emission - price) / emission
            weight = self.weight_dict.get(netuid, 1)
            boost = weight * self.drive
            score = (emission * (1 + boost) - price) / (emission * (1 + boost))
            name = str(subnet.subnet_name) if hasattr(subnet, "subnet_name") else ""
            volume = (
                float(subnet.subnet_volume) if hasattr(subnet, "subnet_volume") else 0.0
            )
            stats[netuid] = SubnetStats(
                price=price,
                emission=emission,
                raw_yield=raw_yield,
                weight=weight,
                boost=1 + boost,
                score=score,
                name=name,
                volume=volume,
            )
        return stats, rank_dict

    async def fetch_validator_stats(
        self, validator: str, subnet: int
    ) -> ValidatorStats:
        """
        Fetch actual metrics for a given validator and subnet.
        Replace the dummy values with actual API calls if available.
        """
        try:
            # Simulate async call to fetch metrics. Replace with real API call.
            await asyncio.sleep(0.01)  # simulate network delay
            # Example dummy values, replace with real data:
            stake = 100.0
            rewards = 10.0
            uptime = 0.98
            # Compute performance_score using dummy factors.
            performance_score = (
                uptime * 0.5 + (rewards / stake if stake > 0 else 0) * 0.5
            )
            return ValidatorStats(
                address=validator,
                stake=stake,
                rewards=rewards,
                uptime=uptime,
                performance_score=performance_score,
            )
        except Exception as e:
            logger.error(
                f"Failed to fetch stats for validator {validator} on subnet {subnet}: {e}"
            )
            return ValidatorStats(
                address=validator, stake=0, rewards=0, uptime=0, performance_score=0
            )

    async def compute_validator_ranking(
        self, subnet: int, stats: SubnetStats
    ) -> Dict[str, float]:
        """
        Compute performance metric for each validator.
        Fetch actual validator metrics and combine with subnet stats.
        """
        validators = self.validators_mapping.get(subnet, [])
        if not validators:
            logger.warning(f"No validators configured for subnet {subnet}")
            return {}

        # Fetch metrics concurrently for all validators in this subnet.
        tasks = [
            self.fetch_validator_stats(validator, subnet) for validator in validators
        ]
        results = await asyncio.gather(*tasks)
        validator_metrics = {}
        for result in results:
            # Combine validator performance and subnet stats.
            # For example, mix subnet.score (factor 0.4) with validator uptime (0.3)
            # and reward rate (rewards/stake, 0.3)
            reward_rate = (result.rewards / result.stake) if result.stake > 0 else 0
            metric = 0.4 * stats.score + 0.3 * result.uptime + 0.3 * reward_rate
            validator_metrics[result.address] = metric

        return validator_metrics

    async def stake_operations(self) -> None:
        logger.info("Starting stake_operations: fetching subnet stats.")
        stats, rank_dict = await self.get_subnet_stats()
        logger.info(f"Fetched stats for {len(stats)} subnets.")

        valid_subnets = [netuid for netuid in self.allowed_subnets if netuid in stats]
        logger.info(f"Valid subnets after filtering: {valid_subnets}")
        if not valid_subnets:
            logger.warning("No allowed subnets with valid stats found.")
            return

        combined_metric = {}
        best_validators = {}  # subnet -> (best_validator, metric)
        # For each subnet, first rank available validators.
        for netuid in valid_subnets:
            # Use the subnet score and other metrics for ranking.
            subnet_stat = stats[netuid]
            # Get validator performance metrics for this subnet.
            validator_metrics = await self.compute_validator_ranking(
                netuid, subnet_stat
            )
            if not validator_metrics:
                continue
            # Pick the best validator in this subnet.
            best_validator = max(validator_metrics, key=lambda k: validator_metrics[k])
            best_metric = validator_metrics[best_validator]
            best_validators[netuid] = (best_validator, best_metric)
            # Combine with subnet metrics. Here we average the subnet score with validator performance.
            combined_metric[netuid] = (subnet_stat.score + best_metric) / 2

        # Rank all best validators per subnet.
        top_n = 3
        sorted_subnets = sorted(
            combined_metric, key=lambda net: combined_metric[net], reverse=True
        )
        selected_subnets = sorted_subnets[: min(top_n, len(sorted_subnets))]
        logger.info(
            f"Selected subnets for staking based on combined validator metrics: {selected_subnets}"
        )

        # Allocate investments: 50%, 30%, 20%
        allocation_ratios = [0.5, 0.3, 0.2]
        total_investment = self.amount_staked
        for idx, netuid in enumerate(selected_subnets):
            if idx >= len(allocation_ratios):
                logger.warning(f"No allocation ratio for subnet {netuid}, skipping")
                continue

            invest_amount = total_investment * allocation_ratios[idx]
            if netuid not in best_validators:
                logger.warning(f"No best validator found for subnet {netuid}, skipping")
                continue

            best_validator, metric = best_validators[netuid]
            if metric <= 0:
                logger.warning(f"Zero or negative metric for subnet {netuid}, skipping")
                continue

            logger.info(
                f"Staking {invest_amount:.4f} TAO to subnet {netuid} for validator {best_validator}"
            )
            wallet_balance = float(
                await self.sub.get_balance(address=self.wallet.coldkey.ss58_address)
            )
            if wallet_balance < invest_amount:
                logger.error(
                    f"Insufficient balance ({wallet_balance:.4f} TAO) to stake {invest_amount:.4f} TAO on subnet {netuid} with validator {best_validator}."
                )
                continue
            try:
                await self.sub.add_stake(
                    wallet=self.wallet,
                    hotkey_ss58=best_validator,
                    netuid=netuid,
                    amount=bt.Balance.from_tao(invest_amount),
                    wait_for_inclusion=True,
                    wait_for_finalization=True,
                )
                logger.info(
                    f"Staked {invest_amount:.4f} TAO to subnet {netuid} for validator {best_validator}"
                )
                self.total_allocated += invest_amount
            except Exception as e:
                logger.error(
                    f"Failed to stake on subnet {netuid} with validator {best_validator}: {e}"
                )

        worst_subnet = min(valid_subnets, key=lambda x: stats[x].score)
        if worst_subnet in best_validators:
            worst_validator = best_validators[worst_subnet][0]
        else:
            logger.warning(f"No validator info for worst subnet {worst_subnet}")
            return

        logger.info(
            f"Worst subnet determined for unstaking: {worst_subnet} with score {stats[worst_subnet].score:.4f}"
        )

        logger.info("Proceeding with unstaking operation.")
        stake_info_before = await self.sub.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=list(best_validators.values())[0][0] if best_validators else "",
            coldkey_ss58=self.wallet.coldkey.ss58_address,
            netuids=self.allowed_subnets,
        )
        worst_stake_obj = stake_info_before.get(worst_subnet)
        if worst_stake_obj is not None:
            current_stake = float(worst_stake_obj.stake)
            logger.info(
                f"Current stake on worst subnet {worst_subnet}: {current_stake}"
            )
            if current_stake > 0:
                price_worst = stats[worst_subnet].price
                unstake_target = self.amount_unstaked / price_worst
                unstake_amt = min(current_stake, unstake_target)
                if unstake_amt > 0:
                    try:
                        await self.sub.unstake(
                            wallet=self.wallet,
                            hotkey_ss58=(
                                list(best_validators.values())[0][0]
                                if best_validators
                                else ""
                            ),
                            netuid=worst_subnet,
                            amount=bt.Balance.from_tao(unstake_amt),
                            wait_for_inclusion=True,
                            wait_for_finalization=True,
                        )
                        logger.info(
                            f"Unstaked {unstake_amt:.4f} stake units from subnet {worst_subnet} "
                            f"(approx. {self.amount_unstaked:.4f} TAO value)"
                        )
                    except Exception as e:
                        logger.error(
                            f"Failed to unstake from subnet {worst_subnet}: {e}"
                        )
                else:
                    logger.info(
                        f"Unstake amount computed as zero for worst subnet {worst_subnet}."
                    )
            else:
                logger.info(f"No stake on worst subnet {worst_subnet} to unstake.")
        else:
            logger.info(
                f"No stake info for worst subnet {worst_subnet}; nothing to unstake."
            )

        stake_info = await self.sub.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=list(best_validators.values())[0][0] if best_validators else "",
            coldkey_ss58=self.wallet.coldkey.ss58_address,
            netuids=self.allowed_subnets,
        )
        balance = float(
            await self.sub.get_balance(address=self.wallet.coldkey.ss58_address)
        )
        logger.info(f"Updated wallet balance: {balance:.4f} TAO")
        self.print_table_rich(
            stake_info, stats, rank_dict, balance, best_validators
        )  # Add best_validators here
        logger.info("Stake operation cycle complete. Waiting for the next block...")
        await self.sub.wait_for_block()

    async def run(self) -> None:
        logger.info("Initializing subtensor connection...")
        self.sub = await get_async_subtensor("finney")
        logger.info("Subtensor connection established.")
        self.wallet = bt.wallet(name=self.wallet_name)
        logger.info(f"Wallet {self.wallet_name} loaded; verifying existence.")
        self.wallet.create_if_non_existent()
        self.wallet.unlock_coldkey()
        logger.info("Wallet unlocked and ready for transactions.")
        try:
            while True:
                try:
                    logger.info("Starting a new stake operation cycle.")
                    await self.stake_operations()
                except Exception as e:
                    logger.error(f"Error in stake loop: {e}")
                    await asyncio.sleep(12)
        finally:
            if self.sub:
                logger.info("Closing subtensor connection.")
                await self.sub.close()


async def main() -> None:
    parser = argparse.ArgumentParser(description="Optimize staking allocations")
    parser.add_argument(
        "--config", default="config.yaml", help="Path to the configuration file"
    )
    args = parser.parse_args()
    logger.info(f"Starting script with config: {args.config}")
    optimizer = StakingOptimizer(args.config)
    await optimizer.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        raise
