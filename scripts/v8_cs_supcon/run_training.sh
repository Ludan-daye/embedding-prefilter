#!/bin/bash
cd "$(dirname "$0")/../.."
export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
python3 scripts/v8_cs_supcon/train_from_cache.py
