#!/bin/bash

# Text-to-Audiobook Service Manager
# Manages Docker services and runs validation checks

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
COMPOSE_FILE="docker-compose.yml"
PROJECT_NAME="text-to-audiobook"

function print_header() {
    echo -e "${BLUE}"
    echo "=============================================="
    echo "  Text-to-Audiobook Service Manager"
    echo "=============================================="
    echo -e "${NC}"
}

function print_usage() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start       Start all services"
    echo "  stop        Stop all services"
    echo "  restart     Restart all services"
    echo "  status      Show service status"
    echo "  validate    Run service validation checks"
    echo "  logs        Show service logs"
    echo "  clean       Stop and remove all containers/volumes"
    echo "  help        Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start"
    echo "  $0 validate"
    echo "  $0 logs kafka"
}

function start_services() {
    echo -e "${GREEN}🚀 Starting Text-to-Audiobook services...${NC}"
    
    # Check if docker-compose file exists
    if [ ! -f "$COMPOSE_FILE" ]; then
        echo -e "${RED}❌ Error: $COMPOSE_FILE not found${NC}"
        exit 1
    fi
    
    # Start services
    docker-compose -f "$COMPOSE_FILE" up -d
    
    echo -e "${GREEN}✅ Services started successfully${NC}"
    echo ""
    echo -e "${YELLOW}⏳ Waiting for services to initialize...${NC}"
    sleep 10
    
    echo -e "${BLUE}📊 Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo -e "${YELLOW}💡 Run '$0 validate' to check service health${NC}"
}

function stop_services() {
    echo -e "${YELLOW}🛑 Stopping Text-to-Audiobook services...${NC}"
    docker-compose -f "$COMPOSE_FILE" down
    echo -e "${GREEN}✅ Services stopped successfully${NC}"
}

function restart_services() {
    echo -e "${YELLOW}🔄 Restarting Text-to-Audiobook services...${NC}"
    stop_services
    sleep 3
    start_services
}

function show_status() {
    echo -e "${BLUE}📊 Service Status:${NC}"
    docker-compose -f "$COMPOSE_FILE" ps
    
    echo ""
    echo -e "${BLUE}💾 Resource Usage:${NC}"
    docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}" $(docker-compose -f "$COMPOSE_FILE" ps -q)
}

function validate_services() {
    echo -e "${BLUE}🔍 Running service validation...${NC}"
    
    # Check if validation script exists
    if [ ! -f "validate_services.py" ]; then
        echo -e "${RED}❌ Error: validate_services.py not found${NC}"
        exit 1
    fi
    
    # Run validation
    python3 validate_services.py
}

function show_logs() {
    local service="$1"
    
    if [ -z "$service" ]; then
        echo -e "${BLUE}📝 Showing logs for all services (last 100 lines):${NC}"
        docker-compose -f "$COMPOSE_FILE" logs --tail=100
    else
        echo -e "${BLUE}📝 Showing logs for $service:${NC}"
        docker-compose -f "$COMPOSE_FILE" logs --tail=100 -f "$service"
    fi
}

function clean_services() {
    echo -e "${YELLOW}🧹 Cleaning up Text-to-Audiobook services...${NC}"
    echo -e "${RED}⚠️  This will remove all containers, networks, and volumes!${NC}"
    
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker-compose -f "$COMPOSE_FILE" down -v --rmi local
        docker system prune -f
        echo -e "${GREEN}✅ Cleanup completed${NC}"
    else
        echo -e "${YELLOW}❌ Cleanup cancelled${NC}"
    fi
}

function check_prerequisites() {
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        echo -e "${RED}❌ Error: Docker is not running${NC}"
        exit 1
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        echo -e "${RED}❌ Error: docker-compose is not installed${NC}"
        exit 1
    fi
}

# Main script logic
print_header

# Check prerequisites
check_prerequisites

# Handle commands
case "${1:-help}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    status)
        show_status
        ;;
    validate)
        validate_services
        ;;
    logs)
        show_logs "$2"
        ;;
    clean)
        clean_services
        ;;
    help|--help|-h)
        print_usage
        ;;
    *)
        echo -e "${RED}❌ Unknown command: $1${NC}"
        echo ""
        print_usage
        exit 1
        ;;
esac