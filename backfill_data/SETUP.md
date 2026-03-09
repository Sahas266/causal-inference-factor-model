# Quick Setup Guide

## 🚀 Getting Started

This guide will help you set up and run the Multi-Provider Data Backfill System in under 10 minutes.

## Step 1: Install Dependencies

```bash
cd backfill_data
pip install -r requirements.txt
```

## Step 2: Set Up Environment Variables

Create a `.env` file with your credentials:

```bash
# Copy the example (you'll need to manually create .env)
cat > .env << 'EOF'
# Supabase Configuration
SUPABASE_URL=https://your-project-id.supabase.co
SUPABASE_KEY=your-service-role-key-here

# CoinMetrics API Key
COINMETRICS_API_KEY=your-coinmetrics-api-key-here

# Dune API Key
DUNE_API_KEY=your-dune-api-key-here

# Optional
LOG_LEVEL=INFO
MAX_WORKERS=10
EOF
```

**Important**: Replace the placeholder values with your actual credentials!

### Getting Your Credentials

**Supabase**:
1. Go to https://supabase.com/dashboard
2. Select your project
3. Settings → API → Project URL (SUPABASE_URL)
4. Settings → API → service_role key (SUPABASE_KEY)

**CoinMetrics**:
1. Go to https://coinmetrics.io/
2. Sign up for an account
3. Get your API key from the dashboard

**CoinMetrics Community API**:
- Base URL: `https://community-api.coinmetrics.io/v4`
- Community endpoints do not require an API key

## Step 3: Set Up Database

Run the database schema in your Supabase SQL editor:

1. Open Supabase Dashboard → SQL Editor
2. Copy contents from `config/database_schema.sql`
3. Execute the SQL

Or use psql:

```bash
psql -h db.your-project.supabase.co -U postgres -d postgres -f config/database_schema.sql
```

## Step 4: Verify Setup

Test that everything is configured correctly:

```bash
# List available providers
python backfill.py --list-providers

# Should output:
# Available Providers:
# ------------------------------------------
#   • coinmetrics
```

## Step 5: Run Your First Backfill

### Option A: Validate Configuration (Recommended First Step)

```bash
python backfill.py --config config/endpoints/btc_metrics.json --validate-only
```

This will check that:
- API keys are valid
- Endpoints are accessible
- Data is available for the requested date range

### Option B: Run Actual Backfill

```bash
python backfill.py --config config/endpoints/btc_metrics.json
```

This will fetch Bitcoin metrics from CoinMetrics and store them in your Supabase database.

### Option C: Run Provider-Specific Script

```bash
# CoinMetrics
python scripts/backfill_coinmetrics.py --validate-only

# DeFi Llama
python scripts/backfill_defillama.py --validate-only

# CoinGecko and Dune script entry points
python scripts/backfill_coingecko.py --list-endpoints
python scripts/backfill_dune.py --list-endpoints
```

## Step 6: Monitor Progress

### Check Database

```sql
-- See backfill progress
SELECT 
    endpoint_id, 
    provider, 
    status, 
    total_records_fetched,
    last_successful_time,
    updated_at
FROM backfill_progress
ORDER BY updated_at DESC;

-- See data
SELECT 
    asset,
    metric,
    time,
    value
FROM asset_metrics
WHERE asset = 'btc'
ORDER BY time DESC
LIMIT 10;
```

### Check Logs

```bash
# With debug logging
python backfill.py --config config/endpoints/btc_metrics.json --log-level DEBUG
```

## Common Issues

### Issue: "SUPABASE_URL environment variable not set"

**Solution**: Ensure your `.env` file exists and contains valid values.

```bash
# Check .env exists
ls -la .env

# Check values are loaded
python -c "from dotenv import load_dotenv; import os; load_dotenv(); print(os.getenv('SUPABASE_URL'))"
```

### Issue: "Unknown provider: 'coinmetrics'"

**Solution**: The provider registry might not have initialized. Check imports:

```bash
python -c "from src.providers.registry import ProviderRegistry; ProviderRegistry.auto_discover(); print(ProviderRegistry.list_providers())"
```

### Issue: Rate limit errors

**Solution**: Reduce concurrent workers:

```bash
python backfill.py --config config/endpoints/ --max-workers 3
```

## Next Steps

### 1. Backfill More Assets

Edit `config/endpoints/eth_metrics.json` or create new endpoint configs:

```bash
python backfill.py --config config/endpoints/eth_metrics.json
```

### 2. Backfill All Endpoints

```bash
python backfill.py --config config/endpoints/
```

### 3. Resume Failed Backfills

If a backfill fails, you can resume it:

```bash
python backfill.py --resume-failed
```

### 4. Add More Providers

See the [README.md](README.md#adding-a-new-provider) for instructions on adding Glassnode, Messari, etc.

## Testing

Run the test suite to verify everything works:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test
pytest tests/providers/test_coinmetrics.py -v
```

## Production Checklist

Before running in production:

- [ ] Database indexes created (via database_schema.sql)
- [ ] Environment variables set in production environment
- [ ] Appropriate `MAX_WORKERS` configured for your API tier
- [ ] Monitoring and alerting set up for failed backfills
- [ ] Log aggregation configured
- [ ] Backup strategy in place for database
- [ ] Rate limits tuned based on your API tier

## Support

- Check [README.md](README.md) for detailed documentation
- Review [project_goals.md](project_goals.md) for architecture details
- Review test files in `tests/` for usage examples

## Example Session

```bash
# Complete setup example
cd backfill_data
pip install -r requirements.txt

# Create .env with your credentials
nano .env

# Test configuration
python backfill.py --list-providers

# Validate endpoint
python backfill.py --config config/endpoints/btc_metrics.json --validate-only

# Run backfill with debug logging
python backfill.py --config config/endpoints/btc_metrics.json --log-level DEBUG

# Check database
psql $SUPABASE_URL -c "SELECT COUNT(*) FROM asset_metrics;"

# Success! 🎉
```

## Congratulations! 🎉

You've successfully set up the Multi-Provider Data Backfill System. The system will now:

- ✅ Automatically handle rate limiting
- ✅ Resume from checkpoints on failure
- ✅ Support adding new providers without code changes
- ✅ Scale to millions of records with efficient batching
- ✅ Provide comprehensive logging and monitoring

Happy backfilling!

