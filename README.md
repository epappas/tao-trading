# TAO Trading

A Python script to optimize staking allocations across subnets using TAO.

## Overview

This script:

- Reads configuration from a YAML file.
- Computes subnet statistics.
- Stakes to the best subnet and unstakes from the worst based on a hysteresis threshold.
- Displays results using a Rich table.

## Algorithm Approach

The algorithm computes performance scores for each subnet based on factors like emission, price, yield, and a boost factor. It then identifies the best and worst performing subnets. By applying a hysteresis threshold, it prevents excessive trading by acting only when the performance difference is significant. This approach helps to optimize staking allocations while reducing transaction frequency.

## Usage

Ensure you have [uv installed](https://docs.astral.sh/uv/getting-started/installation/) before running the script.

Run the script with:

```bash
./tao-trading.py
```

Ensure `config.yaml` and the ranks file are properly configured.

## License

This project is licensed under the MIT License. See the [LICENSE](../LICENSE) file for details.
