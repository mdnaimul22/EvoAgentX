import asyncio
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from pathlib import Path
import aiohttp
from aiohttp import web, WSMsgType
import aiohttp_cors
import aiofiles
import jinja2
import sqlite3

class WebDashboard:
    
    def __init__(self, ecosystem):
        self.ecosystem = ecosystem
        self.logger = ecosystem.logger
        self.config = ecosystem.config
        self.app = web.Application()
        self.websockets = set()
        
        self.setup_templates()
        
        self.setup_routes()
        
        self.setup_cors()
        
    def setup_templates(self):
        template_dir = Path(__file__).parent / "templates"
        template_dir.mkdir(exist_ok=True)
        
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(template_dir)),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )
        
        self.create_default_templates()
    
    def create_default_templates(self):
        template_dir = Path(__file__).parent / "templates"
        
        dashboard_html = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ecosystem Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        .metric-card { transition: all 0.3s ease; }
        .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.15); }
        .agent-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
        .evolution-card { background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); }
        .knowledge-card { background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); }
        .health-card { background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); }
    </style>
</head>
<body class="bg-gray-100 min-h-screen">
    <div class="container mx-auto px-4 py-8">
        <header class="mb-8">
            <h1 class="text-4xl font-bold text-gray-800 mb-2">🤖 Ecosystem Dashboard</h1>
            <p class="text-gray-600">Real-time monitoring of autonomous agent ecosystem</p>
            <div class="mt-4 flex items-center space-x-4">
                <span class="inline-flex items-center px-3 py-1 rounded-full text-sm font-medium bg-green-100 text-green-800">
                    <span class="w-2 h-2 bg-green-400 rounded-full mr-2"></span>
                    System Online
                </span>
                <span class="text-sm text-gray-500">Uptime: <span id="uptime">{{ uptime }}</span></span>
            </div>
        </header>
        <div class="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
            <div class="metric-card agent-card rounded-lg p-6 text-white">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white/80 text-sm">Active Agents</p>
                        <p class="text-3xl font-bold" id="agent-count">{{ agent_count }}</p>
                    </div>
                    <div class="text-4xl opacity-80">🤖</div>
                </div>
            </div>
            
            <div class="metric-card evolution-card rounded-lg p-6 text-white">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white/80 text-sm">Intelligence Score</p>
                        <p class="text-3xl font-bold" id="intelligence-score">{{ intelligence_score }}</p>
                    </div>
                    <div class="text-4xl opacity-80">🧠</div>
                </div>
            </div>
            
            <div class="metric-card knowledge-card rounded-lg p-6 text-white">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white/80 text-sm">Knowledge Domains</p>
                        <p class="text-3xl font-bold" id="knowledge-domains">{{ knowledge_domains }}</p>
                    </div>
                    <div class="text-4xl opacity-80">📚</div>
                </div>
            </div>
            
            <div class="metric-card health-card rounded-lg p-6 text-white">
                <div class="flex items-center justify-between">
                    <div>
                        <p class="text-white/80 text-sm">System Health</p>
                        <p class="text-3xl font-bold" id="system-health">{{ system_health }}</p>
                    </div>
                    <div class="text-4xl opacity-80">💚</div>
                </div>
            </div>
        </div>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold mb-4">Intelligence Evolution</h3>
                <canvas id="intelligence-chart" width="400" height="200"></canvas>
            </div>
            
            <div class="bg-white rounded-lg shadow-md p-6">
                <h3 class="text-lg font-semibold mb-4">Agent Distribution</h3>
                <canvas id="agent-distribution-chart" width="400" height="200"></canvas>
            </div>
        </div>
        <div class="bg-white rounded-lg shadow-md p-6 mb-8">
            <h3 class="text-lg font-semibold mb-4">Active Agents</h3>
            <div class="overflow-x-auto">
                <table class="min-w-full table-auto">
                    <thead>
                        <tr class="bg-gray-50">
                            <th class="px-4 py-2 text-left">Name</th>
                            <th class="px-4 py-2 text-left">Role</th>
                            <th class="px-4 py-2 text-left">Intelligence</th>
                            <th class="px-4 py-2 text-left">Energy</th>
                            <th class="px-4 py-2 text-left">Last Active</th>
                        </tr>
                    </thead>
                    <tbody id="agents-table">
                    </tbody>
                </table>
            </div>
        </div>
        <div class="bg-white rounded-lg shadow-md p-6">
            <h3 class="text-lg font-semibold mb-4">System Logs</h3>
            <div class="bg-gray-900 text-green-400 p-4 rounded font-mono text-sm h-64 overflow-y-auto" id="logs-container">
            </div>
        </div>
    </div>
    <script>
        const ws = new WebSocket(`ws://${window.location.host}/ws`);
        
        ws.onmessage = function(event) {
            const data = JSON.parse(event.data);
            updateDashboard(data);
        };
        
        ws.onopen = function() {
            console.log('WebSocket connected');
        };
        
        ws.onclose = function() {
            console.log('WebSocket disconnected');
            setTimeout(() => location.reload(), 5000);
        };
        
        const intelligenceCtx = document.getElementById('intelligence-chart').getContext('2d');
        const intelligenceChart = new Chart(intelligenceCtx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Intelligence Score',
                    data: [],
                    borderColor: 'rgb(75, 192, 192)',
                    backgroundColor: 'rgba(75, 192, 192, 0.2)',
                    tension: 0.1
                }]
            },
            options: {
                responsive: true,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });
        
        const distributionCtx = document.getElementById('agent-distribution-chart').getContext('2d');
        const distributionChart = new Chart(distributionCtx, {
            type: 'doughnut',
            data: {
                labels: [],
                datasets: [{
                    data: [],
                    backgroundColor: [
                        '#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0',
                        '#9966FF', '#FF9F40', '#FF6384', '#C9CBCF'
                    ]
                }]
            },
            options: {
                responsive: true
            }
        });
        
        function updateDashboard(data) {
            document.getElementById('agent-count').textContent = data.agent_count;
            document.getElementById('intelligence-score').textContent = data.intelligence_score.toFixed(3);
            document.getElementById('knowledge-domains').textContent = data.knowledge_domains;
            document.getElementById('system-health').textContent = data.system_health.toFixed(3);
            document.getElementById('uptime').textContent = data.uptime;
            
            if (data.intelligence_history) {
                intelligenceChart.data.labels = data.intelligence_history.labels;
                intelligenceChart.data.datasets[0].data = data.intelligence_history.data;
                intelligenceChart.update();
            }
            
            if (data.agent_distribution) {
                distributionChart.data.labels = data.agent_distribution.labels;
                distributionChart.data.datasets[0].data = data.agent_distribution.data;
                distributionChart.update();
            }
            
            if (data.agents) {
                updateAgentsTable(data.agents);
            }
            
            if (data.logs) {
                updateLogs(data.logs);
            }
        }
        
        function updateAgentsTable(agents) {
            const tbody = document.getElementById('agents-table');
            tbody.innerHTML = '';
            
            agents.forEach(agent => {
                const row = document.createElement('tr');
                row.className = 'border-b hover:bg-gray-50';
                row.innerHTML = `
                    <td class="px-4 py-2 font-medium">${agent.name}</td>
                    <td class="px-4 py-2">
                        <span class="px-2 py-1 bg-blue-100 text-blue-800 rounded-full text-xs">
                            ${agent.role}
                        </span>
                    </td>
                    <td class="px-4 py-2">${agent.intelligence.toFixed(3)}</td>
                    <td class="px-4 py-2">
                        <div class="w-full bg-gray-200 rounded-full h-2">
                            <div class="bg-green-600 h-2 rounded-full" style="width: ${agent.energy * 100}%"></div>
                        </div>
                    </td>
                    <td class="px-4 py-2 text-sm text-gray-500">${agent.last_active}</td>
                `;
                tbody.appendChild(row);
            });
        }
        
        function updateLogs(logs) {
            const container = document.getElementById('logs-container');
            container.innerHTML = '';
            
            logs.forEach(log => {
                const logLine = document.createElement('div');
                logLine.className = 'mb-1';
                logLine.innerHTML = `<span class="text-gray-500">[${log.timestamp}]</span> <span class="text-yellow-400">${log.level}</span> ${log.message}`;
                container.appendChild(logLine);
            });
            
            container.scrollTop = container.scrollHeight;
        }
        
        fetch('/api/status')
            .then(response => response.json())
            .then(data => updateDashboard(data))
            .catch(error => console.error('Error fetching initial data:', error));
    </script>
</body>
</html>
        """
        
        dashboard_file = template_dir / "dashboard.html"
        if not dashboard_file.exists():
            with open(dashboard_file, 'w') as f:
                f.write(dashboard_html)
    
    def setup_routes(self):
        self.app.router.add_get('/', self.dashboard_handler)
        self.app.router.add_get('/api/status', self.status_api_handler)
        self.app.router.add_get('/api/agents', self.agents_api_handler)
        self.app.router.add_get('/api/metrics', self.metrics_api_handler)
        self.app.router.add_get('/api/logs', self.logs_api_handler)
        self.app.router.add_post('/api/control', self.control_api_handler)
        self.app.router.add_get('/ws', self.websocket_handler)
        
        static_dir = Path(__file__).parent / "static"
        static_dir.mkdir(exist_ok=True)
        self.app.router.add_static('/static/', path=static_dir, name='static')
    
    def setup_cors(self):
        cors = aiohttp_cors.setup(self.app, defaults={
            "*": aiohttp_cors.ResourceOptions(
                allow_credentials=True,
                expose_headers="*",
                allow_headers="*",
                allow_methods="*"
            )
        })
        
        for route in list(self.app.router.routes()):
            cors.add(route)
    
    async def dashboard_handler(self, request):
        try:
            template = self.jinja_env.get_template('dashboard.html')
            
            status_data = await self.get_status_data()
            
            html = template.render(**status_data)
            return web.Response(text=html, content_type='text/html')
            
        except Exception as e:
            self.logger.error(f"Error rendering dashboard: {e}")
            return web.Response(text="Dashboard Error", status=500)
    
    async def status_api_handler(self, request):
        try:
            data = await self.get_status_data()
            return web.json_response(data)
        except Exception as e:
            self.logger.error(f"Error getting status data: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def get_status_data(self):
        state = self.ecosystem.state
        
        uptime_seconds = (datetime.now() - state.start_time).total_seconds()
        uptime_str = str(timedelta(seconds=int(uptime_seconds)))
        
        intelligence_history = self.get_intelligence_history()
        
        agent_distribution = self.get_agent_distribution()
        
        agents_data = self.get_agents_data()
        
        logs_data = await self.get_recent_logs()
        
        return {
            "agent_count": len(state.agents),
            "intelligence_score": state.system_intelligence_score,
            "knowledge_domains": len(state.collective_knowledge.get("knowledge_domains", [])),
            "system_health": state.system_health_score,
            "uptime": uptime_str,
            "autonomy_level": state.autonomy_level,
            "total_tasks": state.total_tasks_completed,
            "emergency_count": state.emergency_count,
            "intelligence_history": intelligence_history,
            "agent_distribution": agent_distribution,
            "agents": agents_data,
            "logs": logs_data
        }
    
    def get_intelligence_history(self):
        history = self.ecosystem.state.evolution_history[-20:]
        
        labels = []
        data = []
        
        for i, record in enumerate(history):
            labels.append(f"Cycle {i+1}")
            data.append(record.get("new_fitness", 0))
        
        return {"labels": labels, "data": data}
    
    def get_agent_distribution(self):
        from collections import defaultdict
        
        role_counts = defaultdict(int)
        for agent in self.ecosystem.state.agents.values():
            role_counts[agent.role.value] += 1
        
        return {
            "labels": list(role_counts.keys()),
            "data": list(role_counts.values())
        }
    
    def get_agents_data(self):
        agents_data = []
        
        for agent in list(self.ecosystem.state.agents.values())[:20]:
            agents_data.append({
                "name": agent.name,
                "role": agent.role.value.replace("_", " ").title(),
                "intelligence": agent.calculate_intelligence_quotient(),
                "energy": agent.energy_level,
                "last_active": agent.last_active.strftime("%Y-%m-%d %H:%M:%S")
            })
        
        return agents_data
    
    async def get_recent_logs(self):
        logs = []
        
        try:
            with sqlite3.connect(self.ecosystem.db_manager.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT timestamp, severity, event_type, event_data 
                    FROM events 
                    ORDER BY timestamp DESC 
                    LIMIT 50
                """)
                
                rows = cursor.fetchall()
                
                for row in rows:
                    logs.append({
                        "timestamp": row[0][:19],
                        "level": row[1],
                        "message": f"{row[2]}: {row[3][:100]}..."
                    })
        
        except Exception as e:
            self.logger.error(f"Error getting logs: {e}")
            logs.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "level": "ERROR",
                "message": f"Error retrieving logs: {str(e)}"
            })
        
        return logs
    
    async def agents_api_handler(self, request):
        try:
            agents_data = []
            
            for agent in self.ecosystem.state.agents.values():
                agents_data.append({
                    "id": agent.agent_id,
                    "name": agent.name,
                    "role": agent.role.value,
                    "capabilities": {cap.value: score for cap, score in agent.capabilities.items()},
                    "knowledge_domains": agent.knowledge_domains,
                    "intelligence": agent.calculate_intelligence_quotient(),
                    "energy": agent.energy_level,
                    "trust_score": agent.trust_score,
                    "collaboration_network": len(agent.collaboration_network),
                    "performance_history": len(agent.performance_history),
                    "created_at": agent.created_at.isoformat(),
                    "last_active": agent.last_active.isoformat()
                })
            
            return web.json_response({"agents": agents_data})
            
        except Exception as e:
            self.logger.error(f"Error getting agents data: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def metrics_api_handler(self, request):
        try:
            state = self.ecosystem.state
            
            metrics = {
                "system_metrics": {
                    "intelligence_score": state.system_intelligence_score,
                    "autonomy_level": state.autonomy_level,
                    "system_health_score": state.system_health_score,
                    "collective_creativity_score": state.collective_creativity_score,
                    "self_preservation_instinct": state.self_preservation_instinct
                },
                "operational_metrics": {
                    "total_tasks_completed": state.total_tasks_completed,
                    "total_knowledge_items": state.total_knowledge_items,
                    "emergency_count": state.emergency_count,
                    "uptime_hours": state.uptime_hours,
                    "evolution_cycles": len(state.evolution_history)
                },
                "resource_metrics": state.resource_usage,
                "agent_metrics": {
                    "total_agents": len(state.agents),
                    "avg_intelligence": sum(a.calculate_intelligence_quotient() for a in state.agents.values()) / len(state.agents) if state.agents else 0,
                    "avg_energy": sum(a.energy_level for a in state.agents.values()) / len(state.agents) if state.agents else 0,
                    "avg_trust": sum(a.trust_score for a in state.agents.values()) / len(state.agents) if state.agents else 0
                }
            }
            
            return web.json_response(metrics)
            
        except Exception as e:
            self.logger.error(f"Error getting metrics: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def logs_api_handler(self, request):
        try:
            limit = int(request.query.get('limit', 100))
            logs = await self.get_recent_logs()
            
            return web.json_response({"logs": logs[:limit]})
            
        except Exception as e:
            self.logger.error(f"Error getting logs: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def control_api_handler(self, request):
        try:
            data = await request.json()
            action = data.get('action')
            
            if action == 'pause':
                self.ecosystem.running = False
                return web.json_response({"status": "paused"})
            
            elif action == 'resume':
                self.ecosystem.running = True
                return web.json_response({"status": "resumed"})
            
            elif action == 'force_evolution':
                asyncio.create_task(self.ecosystem.ecosystem_evolution_cycle())
                return web.json_response({"status": "evolution_triggered"})
            
            elif action == 'force_knowledge_synthesis':
                asyncio.create_task(self.ecosystem.knowledge_synthesis_cycle())
                return web.json_response({"status": "knowledge_synthesis_triggered"})
            
            else:
                return web.json_response({"error": "Unknown action"}, status=400)
                
        except Exception as e:
            self.logger.error(f"Error in control API: {e}")
            return web.json_response({"error": str(e)}, status=500)
    
    async def websocket_handler(self, request):
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        
        self.websockets.add(ws)
        self.logger.info("WebSocket client connected")
        
        try:
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    pass
                elif msg.type == WSMsgType.ERROR:
                    self.logger.error(f'WebSocket error: {ws.exception()}')
        except Exception as e:
            self.logger.error(f"WebSocket error: {e}")
        finally:
            self.websockets.discard(ws)
            self.logger.info("WebSocket client disconnected")
        
        return ws
    
    async def broadcast_update(self, data):
        if not self.websockets:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        for ws in self.websockets:
            try:
                await ws.send_str(message)
            except Exception as e:
                self.logger.error(f"Error sending WebSocket message: {e}")
                disconnected.add(ws)
        
        self.websockets -= disconnected
    
    async def start_dashboard(self):
        try:
            runner = web.AppRunner(self.app)
            await runner.setup()
            
            site = web.TCPSite(
                runner, 
                self.config.dashboard_host, 
                self.config.dashboard_port
            )
            await site.start()
            
            self.logger.info(f"Web dashboard started at http://{self.config.dashboard_host}:{self.config.dashboard_port}")
            
            asyncio.create_task(self.periodic_updates())
            
        except Exception as e:
            self.logger.error(f"Error starting web dashboard: {e}")
            raise
    
    async def periodic_updates(self):
        while True:
            try:
                if self.websockets:
                    data = await self.get_status_data()
                    await self.broadcast_update(data)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                self.logger.error(f"Error in periodic updates: {e}")
                await asyncio.sleep(10)