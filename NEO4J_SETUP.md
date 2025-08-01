# Neo4j Setup Guide for EvoAgentX

This guide helps you resolve Neo4j connection issues and set up the graph database for the EvoAgentX ecosystem.

## Problem Description

The error you encountered:
```
ValueError: Failed to connect to Neo4j: Notification filtering is not supported for the Bolt Protocol Version(4, 4)
```

This occurs because the Neo4j driver version 4.4.13 has compatibility issues with newer Neo4j servers.

## Solutions Implemented

### 1. Updated Neo4j Driver Version ✅

**Files Modified:**
- `requirements.txt`: Updated `neo4j==4.4.13` to `neo4j>=5.0.0`
- `pyproject.toml`: Updated Neo4j dependency to `neo4j>=5.0.0`

### 2. Enhanced Configuration ✅

**File Modified:** `ecosystem_core.py`
- Added proper Neo4j connection parameters
- Implemented fallback mechanism for when Neo4j is unavailable
- Added error handling and graceful degradation

### 3. Created Setup Tools ✅

**New Files:**
- `setup_neo4j.py`: Automated Neo4j setup using Docker
- `neo4j_fallback.py`: In-memory graph store fallback
- `NEO4J_SETUP.md`: This setup guide

## Quick Fix Steps

### Option 1: Update Dependencies (Recommended)

```bash
# Update Neo4j driver
pip install neo4j>=5.0.0

# Or reinstall all dependencies
pip install -r requirements.txt
```

### Option 2: Use Docker Setup Script

```bash
# Run the automated setup
python setup_neo4j.py
```

This script will:
- Start Neo4j 5.15 in Docker
- Configure authentication (neo4j/password)
- Test the connection
- Initialize the database schema

### Option 3: Manual Neo4j Setup

1. **Download Neo4j Desktop or Community Edition**
   - Visit: https://neo4j.com/download/
   - Install Neo4j 5.x (not 4.x)

2. **Start Neo4j**
   - Default ports: 7474 (HTTP), 7687 (Bolt)
   - Set password to `password` or update your `.env` file

3. **Test Connection**
   ```bash
   python -c "from neo4j_fallback import test_neo4j_connection; print('✅ Connected' if test_neo4j_connection() else '❌ Failed')"
   ```

## Environment Configuration

Update your `.env` file with Neo4j settings:

```bash
# Neo4j Configuration
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
NEO4J_DATABASE=neo4j
```

## Fallback Mode

If Neo4j is not available, the system will automatically use an in-memory graph store:

- ✅ No external dependencies
- ✅ Full graph operations supported
- ⚠️ Data not persisted between runs
- ⚠️ Limited to single process

## Verification

Test your setup:

```bash
# Test Neo4j connection
python -c "
from neo4j_fallback import test_neo4j_connection
if test_neo4j_connection():
    print('✅ Neo4j is working!')
else:
    print('⚠️ Using fallback mode')
"

# Run the ecosystem
python main.py
```

## Docker Commands

If using Docker for Neo4j:

```bash
# Start Neo4j
docker run -d \
  --name evoagentx-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:5.15

# Check logs
docker logs evoagentx-neo4j

# Stop Neo4j
docker stop evoagentx-neo4j

# Remove container
docker rm evoagentx-neo4j
```

## Troubleshooting

### Connection Refused
- Ensure Neo4j is running on port 7687
- Check firewall settings
- Verify authentication credentials

### Version Compatibility
- Use Neo4j 5.x with neo4j driver >=5.0.0
- Avoid mixing Neo4j 4.x server with 5.x driver

### Memory Issues
- Increase Neo4j heap size in neo4j.conf:
  ```
  dbms.memory.heap.initial_size=512m
  dbms.memory.heap.max_size=1G
  ```

### Docker Issues
- Ensure Docker is running
- Check port conflicts (7474, 7687)
- Verify volume permissions

## Schema Information

The system creates these Neo4j structures:

**Node Types:**
- `Agent`: AI agents with properties (name, role, capabilities)
- `KnowledgeDomain`: Knowledge areas
- `Capability`: Agent capabilities

**Relationships:**
- `COLLABORATES_WITH`: Agent collaboration
- `EXCHANGES_KNOWLEDGE`: Knowledge sharing
- `HAS_CAPABILITY`: Agent capabilities
- `KNOWS_ABOUT`: Knowledge domains

## Performance Tips

1. **Indexes**: The setup script creates optimal indexes
2. **Memory**: Allocate sufficient heap memory
3. **Connections**: Use connection pooling (handled by driver)
4. **Queries**: Use parameterized queries for better performance

## Support

If you continue to have issues:

1. Check the logs in `example_ecosystem/logs/`
2. Run the system with `--log-level DEBUG`
3. Verify Neo4j server logs
4. Test with the fallback mode first

The system is designed to work with or without Neo4j, so you can continue development even if the graph database is not available.