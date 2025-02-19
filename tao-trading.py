#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = [
#   "rich>=13.9.4",
#   "pyyaml>=6.0.2",
#   "bittensor>=9.0.0",
#   "asyncio>=3.4.3",
# ]
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


# Type definitions
class Config(TypedDict):
    wallet: str
    amount_staked: float
    amount_unstaked: float
    validator: str
    ranks_file: str
    ranking_beta: float
    drive: float
    rebalance_threshold: float  # New parameter for hysteresis


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


# Constant for smoothing volume delta
VOLUME_ALPHA: float = 0.1

# Initialize logging and console
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
            "validator",
            "ranks_file",
            "ranking_beta",
            "drive",
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


def read_ranks_file(filename: str) -> List[int]:
    """Read subnet rankings from the specified YAML file."""
    try:
        with open(filename, "r") as f:
            data = yaml.safe_load(f)
        if "ranks" not in data:
            raise ValueError("Key 'ranks' not found in the ranks file")
        return data["ranks"]
    except Exception as e:
        logger.critical(f"Failed to read ranks file {filename}: {e}")
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

    def __init__(self, config_path: str):
        self.config: Config = read_config(config_path)
        self.wallet_name: str = self.config["wallet"]
        self.amount_staked: float = self.config["amount_staked"]
        self.amount_unstaked: float = self.config["amount_unstaked"]
        self.validator: str = self.config["validator"]
        self.ranks_file: str = self.config["ranks_file"]
        self.ranking_beta: float = self.config["ranking_beta"]
        self.drive: float = self.config["drive"]
        self.rebalance_threshold: float = self.config.get("rebalance_threshold", 0.01)
        self.allowed_subnets: List[int] = read_ranks_file(self.ranks_file)
        self.weight_dict: Dict[int, float] = compute_weights_from_ranks(
            self.allowed_subnets, self.ranking_beta
        )
        self.total_allocated: float = 0.0
        self.sub: Any = None  # Will be set in run()
        self.wallet: Any = None
        # Persist moving average state between blocks.
        self.last_volume: Dict[int, float] = {}
        self.avg_vol_delta: Dict[int, float] = {}

    def print_table_rich(
        self,
        stake_info: Dict,
        stats: Dict[int, SubnetStats],
        rank_dict: Dict[int, int],
        balance: float,
    ) -> None:
        """
        Print a Rich table showing staking allocations.

        Columns: Subnet, Name, Boost, Yield, Score, Vol Delta, Emission,
        Price, Stake, Stake Value, and Rank.
        """
        total_stake_value = 0.0
        total_stake = 0.0

        table = Table(
            title="Staking Allocations",
            header_style="bold white on dark_blue",
            box=box.SIMPLE_HEAVY,
        )
        table.add_column("Subnet", justify="right", style="bright_cyan")
        table.add_column("Name", justify="left", style="white")
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

            table.add_row(
                str(netuid),
                name,
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

    async def stake_operations(self) -> None:
        """
        Compute subnet stats, stake to the best subnet and unstake from the worst subnet,
        then print the current staking allocations.
        """
        stats, rank_dict = await self.get_subnet_stats()
        valid_subnets = [netuid for netuid in self.allowed_subnets if netuid in stats]
        if not valid_subnets:
            logger.warning("No allowed subnets with valid stats found.")
            return

        scores = {netuid: stats[netuid].score for netuid in valid_subnets}
        best_subnet = max(valid_subnets, key=lambda x: scores[x])
        worst_subnet = min(valid_subnets, key=lambda x: scores[x])

        # Check for collision (if best and worst are the same).
        if best_subnet == worst_subnet:
            logger.info("Best and worst subnets are identical; skipping rebalancing.")
            return

        # Hysteresis: only act if the difference is above the threshold.
        if (scores[best_subnet] - scores[worst_subnet]) < self.rebalance_threshold:
            logger.info(
                f"Score difference ({scores[best_subnet] - scores[worst_subnet]:.4f}) "
                f"is below threshold ({self.rebalance_threshold}); skipping rebalancing."
            )
            return

        logger.info(
            f"Chosen best subnet: {best_subnet} (Score: {scores[best_subnet]:.4f})"
        )
        logger.info(
            f"Chosen worst subnet: {worst_subnet} (Score: {scores[worst_subnet]:.4f})"
        )

        # Stake into the best subnet (waiting for inclusion/finalization).
        try:
            await self.sub.add_stake(
                wallet=self.wallet,
                hotkey_ss58=self.validator,
                netuid=best_subnet,
                amount=bt.Balance.from_tao(self.amount_staked),
                wait_for_inclusion=True,
                wait_for_finalization=True,
            )
            logger.info(f"Staked {self.amount_staked:.4f} TAO to subnet {best_subnet}")
            self.total_allocated += self.amount_staked
        except Exception as e:
            logger.error(f"Failed to stake on subnet {best_subnet}: {e}")

        # Unstake from the worst subnet.
        stake_info_before = await self.sub.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=self.validator,
            coldkey_ss58=self.wallet.coldkey.ss58_address,
            netuids=self.allowed_subnets,
        )
        worst_stake_obj = stake_info_before.get(worst_subnet)
        if worst_stake_obj is not None:
            current_stake = float(worst_stake_obj.stake)
            if current_stake > 0:
                price_worst = stats[worst_subnet].price
                unstake_target = self.amount_unstaked / price_worst
                unstake_amt = min(current_stake, unstake_target)
                if unstake_amt > 0:
                    try:
                        await self.sub.unstake(
                            wallet=self.wallet,
                            hotkey_ss58=self.validator,
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

        # Retrieve updated stake info and balance.
        stake_info = await self.sub.get_stake_for_coldkey_and_hotkey(
            hotkey_ss58=self.validator,
            coldkey_ss58=self.wallet.coldkey.ss58_address,
            netuids=self.allowed_subnets,
        )
        balance = float(
            await self.sub.get_balance(address=self.wallet.coldkey.ss58_address)
        )
        self.print_table_rich(stake_info, stats, rank_dict, balance)
        logger.info("Waiting for the next block...")
        await self.sub.wait_for_block()

    async def run(self) -> None:
        """Initialize wallet, subtensor connection and run the staking loop."""
        self.sub = await get_async_subtensor("finney")
        self.wallet = bt.wallet(name=self.wallet_name)
        self.wallet.create_if_non_existent()
        self.wallet.unlock_coldkey()
        logger.info(f"Using wallet: {self.wallet.name}")
        try:
            while True:
                try:
                    await self.stake_operations()
                except Exception as e:
                    logger.error(f"Error in stake loop: {e}")
                    await asyncio.sleep(12)
        finally:
            if self.sub:
                await self.sub.close()


async def main() -> None:
    """Main entry point."""
    optimizer = StakingOptimizer("config.yaml")
    await optimizer.run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Service stopped by user")
    except Exception as e:
        logger.critical(f"Critical error: {e}")
        raise
