# Model Context Protocol (MCP)

## Overview
The Model Context Protocol (MCP) is an open standard that enables AI models to interact with external tools and data sources in a standardized way. It solves the "M x N" problem where every AI application needs to write custom integrations for every data source.

## Key Concepts

### 1. MCP Server
An MCP Server is a lightweight process that exposes:
- **Resources**: Data that can be read (like files, database rows).
- **Tools**: Functions that can be executed (like `git commit`, `send_email`).
- **Prompts**: Pre-defined templates for interaction.

### 2. MCP Client
An MCP Client is the AI application (like Claude Desktop, Trae, or a LangChain Agent) that connects to the server. The client does not need to know the implementation details of the server; it just speaks the MCP protocol.

### 3. Transports
MCP supports multiple transports:
- **Stdio**: Communicating via standard input/output (great for local processes).
- **SSE (Server-Sent Events)**: Communicating over HTTP (great for remote servers).

## Why Use MCP?
- **Portability**: Write a "Google Drive Tool" once, run it in Claude, Trae, and Zed.
- **Security**: Users control which servers connect to their AI.
- **Simplicity**: No need to manage complex authentication flows inside the AI model itself.

## Example
A "Filesystem MCP Server" might expose:
- Tool: `list_directory(path)`
- Tool: `read_file(path)`
- Resource: `file:///{path}`

When a user asks "Summarize the files in my desktop", the AI Client asks the MCP Server to list the files, then reads them, without the AI developer writing specific code for file access.
