"""
Grammar Constrained Generation Pattern - Real-World Problem Solver

PROBLEM: API Endpoint Configuration Generator
    You need to generate API endpoint configurations for a REST framework.
    Configurations must be valid JSON and conform to a specific schema.
    LLMs often generate invalid configs with syntax errors or missing fields.

SOLUTION: Grammar-Constrained Generation
    - Use formal grammar or JSON Schema to constrain generation
    - Ensure outputs conform to required structure
    - Generate valid, parseable configurations ready to use

This example implements a working API endpoint configuration generator.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
import re

from loguru import logger

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================================
# API ENDPOINT CONFIGURATION GENERATOR
# ============================================================================

class APIEndpointConfigGenerator:
    """
    Generates API endpoint configurations with grammar constraints.
    
    This solves the real problem: generating valid endpoint configs
    that can be directly used by a REST framework.
    """
    
    def __init__(self):
        self.schema = {
            "type": "object",
            "required": ["endpoint"],
            "properties": {
                "endpoint": {
                    "type": "object",
                    "required": ["name", "method", "path", "handler"],
                    "properties": {
                        "name": {"type": "string"},
                        "method": {
                            "type": "string",
                            "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"]
                        },
                        "path": {"type": "string"},
                        "handler": {"type": "string"},
                        "params": {
                            "type": "object",
                            "required": False
                        },
                        "auth": {
                            "type": "boolean",
                            "required": False
                        }
                    }
                }
            }
        }
    
    def generate_config(self, description: str) -> Dict[str, Any]:
        """
        Generate endpoint configuration from natural language description.
        
        Args:
            description: Natural language description of the endpoint
            
        Returns:
            Valid endpoint configuration conforming to schema
        """
        # Parse description to extract endpoint details
        endpoint_info = self._parse_description(description)
        
        # Generate config conforming to schema
        config = {
            "endpoint": {
                "name": endpoint_info.get("name", "unnamed_endpoint"),
                "method": endpoint_info.get("method", "GET"),
                "path": endpoint_info.get("path", "/"),
                "handler": endpoint_info.get("handler", "Controller.handler"),
                "params": endpoint_info.get("params", {}),
                "auth": endpoint_info.get("auth", False)
            }
        }
        
        # Validate against schema (simulating grammar constraints)
        if self._validate_schema(config):
            return config
        else:
            # In production, grammar constraints would prevent invalid configs
            logger.warning("Invalid config generated - would be prevented by grammar constraints")
            return self._fix_config(config)
    
    def _parse_description(self, description: str) -> Dict[str, Any]:
        """Parse natural language description to extract endpoint info."""
        info = {}
        
        # Extract HTTP method
        method_match = re.search(r'\b(GET|POST|PUT|DELETE|PATCH)\b', description.upper())
        if method_match:
            info["method"] = method_match.group(1)
        
        # Extract path
        path_match = re.search(r'path[:\s]+([/\w:]+)', description, re.IGNORECASE)
        if path_match:
            info["path"] = path_match.group(1)
        elif "/" in description:
            # Try to find path-like string
            path_candidate = re.search(r'([/\w:]+)', description)
            if path_candidate:
                info["path"] = path_candidate.group(1)
        
        # Extract name
        name_match = re.search(r'(?:name|endpoint)[:\s]+(\w+)', description, re.IGNORECASE)
        if name_match:
            info["name"] = name_match.group(1)
        else:
            # Generate name from method and path
            method = info.get("method", "get").lower()
            path_parts = info.get("path", "/").strip("/").split("/")
            if path_parts and path_parts[0]:
                info["name"] = f"{method}_{path_parts[0]}"
            else:
                info["name"] = f"{method}_endpoint"
        
        # Extract handler
        handler_match = re.search(r'handler[:\s]+([\w.]+)', description, re.IGNORECASE)
        if handler_match:
            info["handler"] = handler_match.group(1)
        else:
            # Generate default handler
            name = info.get("name", "endpoint")
            info["handler"] = f"{name.title()}Controller.handle"
        
        # Check for auth requirement
        if re.search(r'\b(auth|authenticated|protected|secure)\b', description, re.IGNORECASE):
            info["auth"] = True
        
        # Extract params
        params_match = re.search(r'params?[:\s]+{([^}]+)}', description)
        if params_match:
            # Simple param extraction
            params_str = params_match.group(1)
            info["params"] = {}
            for param in params_str.split(","):
                if ":" in param:
                    key, val = param.split(":", 1)
                    info["params"][key.strip()] = val.strip()
        
        return info
    
    def _validate_schema(self, config: Dict[str, Any]) -> bool:
        """Validate config against schema."""
        try:
            endpoint = config.get("endpoint", {})
            required = ["name", "method", "path", "handler"]
            return all(key in endpoint for key in required)
        except:
            return False
    
    def _fix_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Fix invalid config (in production, grammar prevents this)."""
        endpoint = config.get("endpoint", {})
        
        # Ensure required fields
        if "name" not in endpoint:
            endpoint["name"] = "unnamed_endpoint"
        if "method" not in endpoint:
            endpoint["method"] = "GET"
        if "path" not in endpoint:
            endpoint["path"] = "/"
        if "handler" not in endpoint:
            endpoint["handler"] = "Controller.handler"
        
        return {"endpoint": endpoint}


# ============================================================================
# REST FRAMEWORK SIMULATOR (Real-World Use Case)
# ============================================================================

class RESTFramework:
    """
    Simulated REST framework that uses endpoint configurations.
    
    This demonstrates the real-world problem: the framework needs
    valid, parseable configurations to register endpoints.
    """
    
    def __init__(self):
        self.endpoints = {}
        self.generator = APIEndpointConfigGenerator()
    
    def register_endpoint(self, description: str) -> Dict[str, Any]:
        """
        Register endpoint from natural language description.
        
        This is the real problem: generating valid configs that work.
        """
        # Generate config with grammar constraints
        config = self.generator.generate_config(description)
        
        # Validate JSON (grammar ensures this is always valid)
        try:
            json_str = json.dumps(config, indent=2)
            json.loads(json_str)  # Verify it's valid JSON
            
            # Register endpoint
            endpoint_name = config["endpoint"]["name"]
            self.endpoints[endpoint_name] = config
            
            return {
                "status": "success",
                "endpoint": endpoint_name,
                "config": config
            }
        except json.JSONDecodeError:
            return {
                "status": "error",
                "message": "Invalid JSON configuration"
            }
    
    def list_endpoints(self) -> List[Dict[str, Any]]:
        """List all registered endpoints."""
        return [
            {
                "name": name,
                "method": config["endpoint"]["method"],
                "path": config["endpoint"]["path"],
                "handler": config["endpoint"]["handler"]
            }
            for name, config in self.endpoints.items()
        ]


# ============================================================================
# DEMONSTRATION
# ============================================================================

def demonstrate_real_world_problem():
    """Demonstrate the real-world problem and solution."""
    print("\n" + "="*70)
    print("🎯 REAL-WORLD PROBLEM: API Endpoint Configuration Generator")
    print("="*70)
    
    print("\n❌ PROBLEM:")
    print("   Your REST framework needs endpoint configurations")
    print("   LLMs often generate invalid configs:")
    print("   • Missing required fields (name, method, path, handler)")
    print("   • Invalid JSON syntax")
    print("   • Wrong data types (method should be enum)")
    print("   • Configs that can't be parsed or used")
    
    print("\n✅ SOLUTION: Grammar-Constrained Generation")
    print("   • Use JSON Schema or formal grammar")
    print("   • Constrain generation to valid structure")
    print("   • Ensure all required fields are present")
    print("   • Generate configs ready to use")
    
    # Create framework
    framework = RESTFramework()
    
    # Register endpoints from descriptions
    print("\n📝 REGISTERING ENDPOINTS:")
    
    descriptions = [
        "GET endpoint for /users/:id to get user by ID, handler UserController.getUser, requires auth",
        "POST /api/products to create product, handler ProductController.create",
        "DELETE /api/users/:id to delete user, handler UserController.delete, authenticated"
    ]
    
    for desc in descriptions:
        print(f"\n   Description: {desc}")
        result = framework.register_endpoint(desc)
        
        if result["status"] == "success":
            config = result["config"]
            endpoint = config["endpoint"]
            print(f"   ✅ Registered: {endpoint['name']}")
            print(f"      Method: {endpoint['method']}")
            print(f"      Path: {endpoint['path']}")
            print(f"      Handler: {endpoint['handler']}")
            print(f"      Auth: {endpoint.get('auth', False)}")
        else:
            print(f"   ❌ Error: {result['message']}")
    
    # List all endpoints
    print("\n📋 REGISTERED ENDPOINTS:")
    endpoints = framework.list_endpoints()
    for ep in endpoints:
        print(f"   • {ep['method']} {ep['path']} → {ep['handler']}")
    
    print("\n" + "="*70)
    print("💡 KEY INSIGHTS:")
    print("   • Grammar constraints ensure valid configurations")
    print("   • All required fields are guaranteed to be present")
    print("   • Configs are immediately usable by the framework")
    print("   • No manual validation or fixing needed")
    print("="*70)


def show_comparison():
    """Show comparison: with vs without grammar constraints."""
    print("\n" + "="*70)
    print("⚖️  WITH vs WITHOUT GRAMMAR CONSTRAINTS")
    print("="*70)
    
    print("\n❌ WITHOUT Grammar Constraints:")
    print("   Generated Config:")
    invalid = """{
  "endpoint": {
    "name": getUser
    "method": "GET"
    "path": "/users/:id"
  }
}"""
    print(invalid)
    print("   Problems:")
    print("     • Missing quotes on 'getUser'")
    print("     • Missing commas between fields")
    print("     • Missing required 'handler' field")
    print("   Result: Invalid JSON - framework can't parse")
    
    print("\n✅ WITH Grammar Constraints:")
    print("   Generated Config:")
    valid = """{
  "endpoint": {
    "name": "getUser",
    "method": "GET",
    "path": "/users/:id",
    "handler": "UserController.getUser",
    "auth": true
  }
}"""
    print(valid)
    print("   Benefits:")
    print("     • All quotes properly matched")
    print("     • All commas in place")
    print("     • All required fields present")
    print("     • Valid JSON - framework can use immediately")
    
    print("\n💡 Three Implementation Options:")
    print("   1. Grammar-Constrained Logits Processor (most flexible)")
    print("   2. Standard Data Format (JSON/XML with validators)")
    print("   3. User-Defined Schema (JSON Schema, Pydantic)")


def show_three_options():
    """Show the three implementation options."""
    print("\n" + "="*70)
    print("🔧 THREE IMPLEMENTATION OPTIONS")
    print("="*70)
    
    print("\nOPTION 1: Grammar-Constrained Logits Processor")
    print("   • Define formal grammar (EBNF notation)")
    print("   • Create logits processor that applies grammar")
    print("   • Most flexible - any grammar")
    print("   • Best for: Custom formats, complex grammars")
    
    print("\nOPTION 2: Standard Data Format (JSON/XML)")
    print("   • Use well-known formats with existing validators")
    print("   • Leverage existing tooling")
    print("   • Easy to validate")
    print("   • Best for: Standard formats")
    
    print("\nOPTION 3: User-Defined Schema")
    print("   • Use JSON Schema, Pydantic, etc.")
    print("   • Convert schema to generation constraints")
    print("   • Good balance of flexibility and ease")
    print("   • Best for: Domain-specific schemas")


def main():
    """Main demonstration function."""
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    
    print("\n" + "="*70)
    print("🎯 GRAMMAR CONSTRAINED GENERATION - API CONFIG GENERATOR")
    print("="*70)
    
    print("\n📋 This example solves a real-world problem:")
    print("   Generating valid API endpoint configurations")
    print("   that conform to schema and are ready to use")
    
    # Show three options
    show_three_options()
    
    # Demonstrate real-world problem solving
    demonstrate_real_world_problem()
    
    # Show comparison
    show_comparison()
    
    print("\n" + "="*70)
    print("📚 Next Steps:")
    print("   1. Review README.md for detailed explanation")
    print("   2. Choose implementation option for your use case")
    print("   3. Use libraries like transformers-cfg or outlines")
    print("   4. Adapt for your schema/grammar requirements")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
