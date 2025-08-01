#!/usr/bin/env python3
"""
Neo4j Setup Script for EvoAgentX
================================

This script helps set up Neo4j for the EvoAgentX ecosystem.
It provides options to:
1. Start Neo4j using Docker
2. Test the connection
3. Initialize the database schema
"""

import os
import subprocess
import sys
from pathlib import Path
import time

def check_docker():
    """Check if Docker is available"""
    try:
        subprocess.run(["docker", "--version"], check=True, capture_output=True)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False

def start_neo4j_docker():
    """Start Neo4j using Docker"""
    print("🐳 Starting Neo4j with Docker...")
    
    # Create data directory
    data_dir = Path("./example_ecosystem/neo4j_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = "neo4j:5.23.0"

    # Docker command to start Neo4j
    docker_cmd = [
        "docker", "run", "-d",
        "--name", "evoagentx-neo4j",
        "-p", "7474:7474",
        "-p", "7687:7687",
        "-e", "NEO4J_AUTH=neo4j/password",
        "-e", "NEO4J_PLUGINS=[\"apoc\"]",
        "-v", f"{data_dir.absolute()}:/data",
        image_name
    ]
    
    try:
        # Pull the Docker image first to show progress
        print(f"🐳 Pulling Neo4j image ({image_name}). This may take a while...")
        subprocess.run(["docker", "pull", image_name], check=True)
        print("✅ Neo4j image pulled successfully.")

        # Remove existing container if it exists
        print("🗑️ Removing existing container if present...")
        subprocess.run(["docker", "rm", "-f", "evoagentx-neo4j"],
                      capture_output=True)
        
        # Start new container
        print("🚀 Starting new Neo4j container...")
        result = subprocess.run(docker_cmd, check=True, capture_output=True, text=True)
        print(f"✅ Neo4j container started: {result.stdout.strip()}")
        
        print("⏳ Waiting for Neo4j to be ready...")
        time.sleep(30)
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Neo4j: {e.stderr}")
        return False
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")
        return False

def test_connection():
    """Test Neo4j connection"""
    print("🔍 Testing Neo4j connection...")
    
    try:
        from neo4j import GraphDatabase
        
        # Connection parameters
        uri = "bolt://localhost:7687"
        username = "neo4j"
        password = "password"
        
        # Test connection
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            result = session.run("RETURN 'Hello, Neo4j!' as message")
            record = result.single()
            print(f"✅ Connection successful: {record['message']}")
            
        driver.close()
        return True
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False

def test_connection_with_retry(max_retries: int, delay: int) -> bool:
    """Test Neo4j connection with a retry mechanism."""
    for i in range(max_retries):
        if test_connection():
            return True
        if i < max_retries - 1:
            print(f"⏳ Retrying connection in {delay} seconds... ({i+1}/{max_retries})")
            time.sleep(delay)
    return False

def initialize_schema():
    """Initialize Neo4j schema for EvoAgentX"""
    print("🏗️  Initializing Neo4j schema...")
    
    try:
        from neo4j import GraphDatabase
        
        uri = "bolt://localhost:7687"
        username = "neo4j"
        password = "password"
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        
        with driver.session() as session:
            # Create constraints and indexes for agents
            schema_queries = [
                "CREATE CONSTRAINT agent_id_unique IF NOT EXISTS FOR (a:Agent) REQUIRE a.agent_id IS UNIQUE",
                "CREATE INDEX agent_role_index IF NOT EXISTS FOR (a:Agent) ON (a.role)",
                "CREATE INDEX agent_name_index IF NOT EXISTS FOR (a:Agent) ON (a.name)",
                
                # Create constraints for knowledge domains
                "CREATE CONSTRAINT knowledge_domain_unique IF NOT EXISTS FOR (k:KnowledgeDomain) REQUIRE k.name IS UNIQUE",
                
                # Create constraints for capabilities
                "CREATE CONSTRAINT capability_unique IF NOT EXISTS FOR (c:Capability) REQUIRE c.name IS UNIQUE",
                
                # Create indexes for relationships
                "CREATE INDEX collaboration_timestamp IF NOT EXISTS FOR ()-[r:COLLABORATES_WITH]-() ON (r.timestamp)",
                "CREATE INDEX knowledge_exchange_timestamp IF NOT EXISTS FOR ()-[r:EXCHANGES_KNOWLEDGE]-() ON (r.timestamp)",
            ]
            
            for query in schema_queries:
                try:
                    session.run(query)
                    print(f"✅ Executed: {query}")
                except Exception as e:
                    print(f"⚠️  Warning for query '{query}': {e}")
        
        driver.close()
        print("✅ Schema initialization completed")
        return True
        
    except Exception as e:
        print(f"❌ Schema initialization failed: {e}")
        return False

def main():
    """Main setup function"""
    print("🤖 EvoAgentX Neo4j Setup")
    print("=" * 40)
    
    # Check if Neo4j driver is installed
    try:
        import neo4j
        print(f"✅ Neo4j driver version: {neo4j.__version__}")
    except ImportError:
        print("❌ Neo4j driver not installed. Please run: pip install neo4j>=5.0.0")
        sys.exit(1)
    
    # Check Docker availability
    if not check_docker():
        print("❌ Docker not available. Please install Docker or set up Neo4j manually.")
        print("Manual setup instructions:")
        print("1. Download Neo4j from https://neo4j.com/download/")
        print("2. Start Neo4j with default settings")
        print("3. Set password to 'password' or update your .env file")
        sys.exit(1)
    
    # Start Neo4j
    if not start_neo4j_docker():
        sys.exit(1)
    
    # Test connection
    # Test connection
    if not test_connection_with_retry(max_retries=12, delay=5):
        print("❌ Could not establish connection to Neo4j after multiple retries.")
        print("Please check the Docker container logs for errors: docker logs evoagentx-neo4j")
        sys.exit(1)
    
    # Initialize schema
    if not initialize_schema():
        sys.exit(1)
    
    print("\n🎉 Neo4j setup completed successfully!")
    print("📊 Neo4j Browser: http://localhost:7474")
    print("🔌 Bolt URI: bolt://localhost:7687")
    print("👤 Username: neo4j")
    print("🔑 Password: password")
    print("\nYou can now run your EvoAgentX ecosystem!")

if __name__ == "__main__":
    main()