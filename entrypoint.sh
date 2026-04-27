#!/usr/bin/env bash
set -e

ls /app
/app/scripts/wandb_login

exec "$@"
