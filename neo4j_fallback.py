import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict

class InMemoryGraphStore:
    
    def __init__(self):
        self.nodes = {}
        self.relationships = defaultdict(list)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Using in-memory graph store (Neo4j fallback)")
    
    def create_node(self, node_id: str, labels: List[str], properties: Dict[str, Any]):
        self.nodes[node_id] = {
            'id': node_id,
            'labels': labels,
            'properties': properties,
            'created_at': datetime.now().isoformat()
        }
        return node_id
    
    def create_relationship(self, from_node: str, to_node: str, 
                          relationship_type: str, properties: Dict[str, Any] = None):
        if from_node not in self.nodes or to_node not in self.nodes:
            raise ValueError(f"Nodes must exist before creating relationship")
        
        relationship = {
            'from': from_node,
            'to': to_node,
            'type': relationship_type,
            'properties': properties or {},
            'created_at': datetime.now().isoformat()
        }
        
        self.relationships[from_node].append(relationship)
        return relationship
    
    def find_nodes(self, label: str = None, properties: Dict[str, Any] = None) -> List[Dict]:
        results = []
        
        for node_id, node_data in self.nodes.items():
            if label and label not in node_data['labels']:
                continue
            
            if properties:
                match = True
                for key, value in properties.items():
                    if node_data['properties'].get(key) != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append(node_data)
        
        return results
    
    def find_relationships(self, from_node: str = None, to_node: str = None, 
                          relationship_type: str = None) -> List[Dict]:
        results = []
        
        for node_id, relationships in self.relationships.items():
            for rel in relationships:
                if from_node and rel['from'] != from_node:
                    continue
                if to_node and rel['to'] != to_node:
                    continue
                if relationship_type and rel['type'] != relationship_type:
                    continue
                
                results.append(rel)
        
        return results
    
    def get_node_relationships(self, node_id: str) -> List[Dict]:
        outgoing = self.relationships.get(node_id, [])
        
        incoming = []
        for other_node, relationships in self.relationships.items():
            for rel in relationships:
                if rel['to'] == node_id:
                    incoming.append(rel)
        
        return outgoing + incoming
    
    def update_node(self, node_id: str, properties: Dict[str, Any]):
        if node_id not in self.nodes:
            raise ValueError(f"Node {node_id} not found")
        
        self.nodes[node_id]['properties'].update(properties)
        self.nodes[node_id]['updated_at'] = datetime.now().isoformat()
    
    def delete_node(self, node_id: str):
        if node_id not in self.nodes:
            return
        
        del self.nodes[node_id]
        
        if node_id in self.relationships:
            del self.relationships[node_id]
        
        for other_node in list(self.relationships.keys()):
            self.relationships[other_node] = [
                rel for rel in self.relationships[other_node] 
                if rel['to'] != node_id
            ]
    
    def get_stats(self) -> Dict[str, int]:
        total_relationships = sum(len(rels) for rels in self.relationships.values())
        
        return {
            'nodes': len(self.nodes),
            'relationships': total_relationships,
            'node_labels': len(set(
                label for node in self.nodes.values() 
                for label in node['labels']
            ))
        }
    
    def export_data(self) -> Dict[str, Any]:
        return {
            'nodes': self.nodes,
            'relationships': dict(self.relationships),
            'exported_at': datetime.now().isoformat()
        }
    
    def import_data(self, data: Dict[str, Any]):
        self.nodes = data.get('nodes', {})
        self.relationships = defaultdict(list, data.get('relationships', {}))
        self.logger.info(f"Imported {len(self.nodes)} nodes and graph data")


def create_fallback_graph_config():
    return {
        'graph_name': 'in_memory',
        'fallback': True,
        'store_instance': InMemoryGraphStore()
    }


def test_neo4j_connection(uri: str = "bolt://103.189.236.237:7687", 
                         username: str = "neo4j", 
                         password: str = "Naimul@neo4j1#") -> bool:
    try:
        from neo4j import GraphDatabase
        
        driver = GraphDatabase.driver(uri, auth=(username, password))
        with driver.session() as session:
            session.run("RETURN 1")
        driver.close()
        return True
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Neo4j not available: {e}")
        return False


def get_graph_store_config(fallback_enabled: bool = True):
    
    if test_neo4j_connection():
        return {
            'graph_name': 'neo4j',
            'uri': 'bolt://103.189.236.237:7687',
            'username': 'neo4j',
            'password': 'Naimul@neo4j1#',
            'database': 'neo4j',
            'fallback': False
        }
    
    if fallback_enabled:
        logging.getLogger(__name__).info("Using in-memory graph store fallback")
        return create_fallback_graph_config()
    
    raise ConnectionError("Neo4j not available and fallback disabled")


if __name__ == "__main__":
    store = InMemoryGraphStore()
    
    store.create_node("agent1", ["Agent"], {"name": "TestAgent", "role": "meta_learning"})
    store.create_node("agent2", ["Agent"], {"name": "TestAgent2", "role": "system_optimization"})
    
    store.create_relationship("agent1", "agent2", "COLLABORATES_WITH", 
                            {"strength": 0.8, "timestamp": datetime.now().isoformat()})
    
    agents = store.find_nodes("Agent")
    print(f"Found {len(agents)} agents")
    
    relationships = store.find_relationships(relationship_type="COLLABORATES_WITH")
    print(f"Found {len(relationships)} collaboration relationships")
    
    stats = store.get_stats()
    print(f"Graph stats: {stats}")