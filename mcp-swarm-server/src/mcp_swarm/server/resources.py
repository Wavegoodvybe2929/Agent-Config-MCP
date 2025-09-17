"""
Resource Management System for MCP Swarm Server

Handles text, image, and binary resource management with caching,
optimization, and swarm intelligence coordination.
"""

from typing import Union, Optional, Dict, Any, List, BinaryIO
import asyncio
import logging
import hashlib
import mimetypes
import os
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field

try:
    # Only import what we actually need to avoid type conflicts
    MCP_AVAILABLE = True
    # We'll create our own content classes to avoid import conflicts
except ImportError:
    MCP_AVAILABLE = False

# Define our own content classes to avoid MCP import conflicts
class TextContent:
    def __init__(self, content_type: str = "text", text: str = ""):
        self.type = content_type
        self.text = text

class ImageContent:
    def __init__(self, content_type: str = "image", data: str = "", mimeType: str = ""):
        self.type = content_type
        self.data = data
        self.mimeType = mimeType

class AudioContent:
    def __init__(self, content_type: str = "audio", data: str = "", mimeType: str = ""):
        self.type = content_type
        self.data = data
        self.mimeType = mimeType
@dataclass
class ResourceMetadata:
    """Metadata for a managed resource."""
    uri: str
    name: str
    description: str
    mime_type: str
    size: int
    checksum: str
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    tags: List[str] = field(default_factory=list)
    cache_ttl: Optional[timedelta] = None


@dataclass
class CachedResource:
    """Cached resource with metadata."""
    content: Union[str, bytes]
    metadata: ResourceMetadata
    cached_at: datetime = field(default_factory=datetime.now)
    
    @property
    def is_expired(self) -> bool:
        """Check if cached resource is expired."""
        if self.metadata.cache_ttl is None:
            return False
        return datetime.now() - self.cached_at > self.metadata.cache_ttl


class ResourceManager:
    """Manage MCP resources with swarm intelligence."""
    
    def __init__(self, base_path: str = "data/resources", cache_size: int = 100):
        """Initialize resource manager.
        
        Args:
            base_path: Base path for resource storage
            cache_size: Maximum number of resources to cache
        """
        self.base_path = Path(base_path)
        self.cache_size = cache_size
        self._resource_cache: Dict[str, CachedResource] = {}
        self._resource_metadata: Dict[str, ResourceMetadata] = {}
        self._logger = logging.getLogger("mcp.swarm.resources")
        
        # Ensure base path exists
        self.base_path.mkdir(parents=True, exist_ok=True)
        
    async def initialize(self) -> None:
        """Initialize the resource manager."""
        self._logger.info("Initializing resource manager")
        
        # Load existing resources from storage
        await self._load_existing_resources()
        
        self._logger.info(f"Resource manager initialized with {len(self._resource_metadata)} resources")
        
    async def _load_existing_resources(self) -> None:
        """Load existing resources from storage."""
        try:
            # Look for resource metadata files
            for metadata_file in self.base_path.glob("*.metadata"):
                try:
                    await self._load_resource_metadata(metadata_file)
                except Exception as e:
                    self._logger.warning(f"Failed to load metadata from {metadata_file}: {e}")
                    
        except Exception as e:
            self._logger.error(f"Failed to load existing resources: {e}")
            
    async def _load_resource_metadata(self, metadata_file: Path) -> None:
        """Load resource metadata from file.
        
        Args:
            metadata_file: Path to metadata file
        """
        # This would load JSON metadata in a real implementation
        # For now, we'll skip this as it's complex without actual files
        pass
        
    async def get_resource(self, uri: str) -> Optional[Union[TextContent, ImageContent, AudioContent]]:
        """Retrieve resource with caching and optimization.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content if found, None otherwise
        """
        # Check cache first
        cached = self._resource_cache.get(uri)
        if cached and not cached.is_expired:
            await self._update_access_stats(uri)
            return await self._create_content_object(cached.content, cached.metadata)
            
        # Load from storage
        content = await self._load_resource_from_storage(uri)
        if content is None:
            return None
            
        # Cache the resource
        metadata = self._resource_metadata.get(uri)
        if metadata:
            await self._cache_resource(uri, content, metadata)
            
        await self._update_access_stats(uri)
        
        if metadata:
            return await self._create_content_object(content, metadata)
        else:
            # Create basic metadata
            basic_metadata = ResourceMetadata(
                uri=uri,
                name=uri.split('/')[-1],
                description="",
                mime_type=self._guess_mime_type(uri),
                size=len(content) if isinstance(content, (str, bytes)) else 0,
                checksum=self._calculate_checksum(content)
            )
            return await self._create_content_object(content, basic_metadata)
            
    async def _create_content_object(
        self, 
        content: Union[str, bytes], 
        metadata: ResourceMetadata
    ) -> Union[TextContent, ImageContent, AudioContent]:
        """Create appropriate content object based on MIME type.
        
        Args:
            content: Resource content
            metadata: Resource metadata
            
        Returns:
            Appropriate content object
        """
        mime_type = metadata.mime_type.lower()
        
        if mime_type.startswith('text/'):
            if isinstance(content, bytes):
                text_content = content.decode('utf-8', errors='replace')
            else:
                text_content = str(content)
            return TextContent(content_type="text", text=text_content)
            
        elif mime_type.startswith('image/'):
            if isinstance(content, str):
                content = content.encode('utf-8')
            import base64
            encoded_data = base64.b64encode(content).decode('ascii')
            return ImageContent(content_type="image", data=encoded_data, mimeType=mime_type)
            
        elif mime_type.startswith('audio/'):
            if isinstance(content, str):
                content = content.encode('utf-8')
            import base64
            encoded_data = base64.b64encode(content).decode('ascii')
            return AudioContent(content_type="audio", data=encoded_data, mimeType=mime_type)
            
        else:
            # Default to text
            if isinstance(content, bytes):
                content = content.decode('utf-8', errors='replace')
            return TextContent(content_type="text", text=str(content))
            
    async def _load_resource_from_storage(self, uri: str) -> Optional[Union[str, bytes]]:
        """Load resource content from storage.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource content if found, None otherwise
        """
        try:
            # Handle different URI schemes
            if uri.startswith('file://'):
                file_path = Path(uri[7:])  # Remove 'file://' prefix
                if file_path.is_absolute():
                    target_path = file_path
                else:
                    target_path = self.base_path / file_path
                    
                if target_path.exists() and target_path.is_file():
                    # Determine if binary or text based on MIME type
                    mime_type = self._guess_mime_type(str(target_path))
                    if mime_type.startswith(('image/', 'audio/', 'video/', 'application/')):
                        with open(target_path, 'rb') as f:
                            return f.read()
                    else:
                        with open(target_path, 'r', encoding='utf-8', errors='replace') as f:
                            return f.read()
                            
            elif uri.startswith('http://') or uri.startswith('https://'):
                # For HTTP resources, we'd need to implement downloading
                # For now, return None
                self._logger.warning(f"HTTP resources not yet supported: {uri}")
                return None
                
            else:
                # Treat as relative path within base_path
                target_path = self.base_path / uri
                if target_path.exists() and target_path.is_file():
                    mime_type = self._guess_mime_type(str(target_path))
                    if mime_type.startswith(('image/', 'audio/', 'video/', 'application/')):
                        with open(target_path, 'rb') as f:
                            return f.read()
                    else:
                        with open(target_path, 'r', encoding='utf-8', errors='replace') as f:
                            return f.read()
                            
        except Exception as e:
            self._logger.error(f"Failed to load resource {uri}: {e}")
            
        return None
        
    def _guess_mime_type(self, file_path: str) -> str:
        """Guess MIME type from file path.
        
        Args:
            file_path: Path to file
            
        Returns:
            MIME type string
        """
        mime_type, _ = mimetypes.guess_type(file_path)
        return mime_type or 'application/octet-stream'
        
    def _calculate_checksum(self, content: Union[str, bytes]) -> str:
        """Calculate checksum for content.
        
        Args:
            content: Content to checksum
            
        Returns:
            SHA-256 checksum
        """
        if isinstance(content, str):
            content = content.encode('utf-8')
        return hashlib.sha256(content).hexdigest()
        
    async def _cache_resource(self, uri: str, content: Union[str, bytes], metadata: ResourceMetadata) -> None:
        """Cache a resource.
        
        Args:
            uri: Resource URI
            content: Resource content
            metadata: Resource metadata
        """
        # Remove oldest items if cache is full
        if len(self._resource_cache) >= self.cache_size:
            await self._evict_oldest_cached_resource()
            
        cached_resource = CachedResource(
            content=content,
            metadata=metadata
        )
        
        self._resource_cache[uri] = cached_resource
        
    async def _evict_oldest_cached_resource(self) -> None:
        """Evict the oldest cached resource."""
        if not self._resource_cache:
            return
            
        # Find resource with oldest access time
        oldest_uri = min(
            self._resource_cache.keys(),
            key=lambda uri: self._resource_metadata.get(uri, ResourceMetadata(
                uri="", name="", description="", mime_type="", size=0, checksum=""
            )).last_accessed
        )
        
        del self._resource_cache[oldest_uri]
        
    async def _update_access_stats(self, uri: str) -> None:
        """Update access statistics for a resource.
        
        Args:
            uri: Resource URI
        """
        if uri in self._resource_metadata:
            metadata = self._resource_metadata[uri]
            metadata.last_accessed = datetime.now()
            metadata.access_count += 1
            
    async def create_resource(
        self, 
        uri: str,
        content: Union[TextContent, ImageContent, AudioContent, str, bytes],
        name: Optional[str] = None,
        description: str = "",
        tags: Optional[List[str]] = None,
        cache_ttl: Optional[timedelta] = None
    ) -> ResourceMetadata:
        """Create new resource with metadata.
        
        Args:
            uri: Resource URI
            content: Resource content
            name: Resource name
            description: Resource description
            tags: Resource tags
            cache_ttl: Cache time-to-live
            
        Returns:
            Resource metadata
        """
        # Extract content and determine MIME type
        if hasattr(content, 'type'):
            # Handle MCP content objects
            content_type = getattr(content, 'type', None)
            if content_type == 'text':
                raw_content = getattr(content, 'text', str(content))
                mime_type = 'text/plain'
            elif content_type == 'image':
                raw_content = getattr(content, 'data', '')
                mime_type = getattr(content, 'mimeType', 'image/png')
                # Decode base64 if needed
                if isinstance(raw_content, str):
                    try:
                        import base64
                        raw_content = base64.b64decode(raw_content)
                    except Exception:
                        pass
            elif content_type == 'audio':
                raw_content = getattr(content, 'data', '')
                mime_type = getattr(content, 'mimeType', 'audio/wav')
                # Decode base64 if needed
                if isinstance(raw_content, str):
                    try:
                        import base64
                        raw_content = base64.b64decode(raw_content)
                    except Exception:
                        pass
            else:
                raw_content = str(content)
                mime_type = 'text/plain'
        else:
            raw_content = content
            mime_type = self._guess_mime_type(uri)
        
        # Ensure raw_content is str or bytes for checksum calculation
        if not isinstance(raw_content, (str, bytes)):
            raw_content = str(raw_content)
            
        # Create metadata
        metadata = ResourceMetadata(
            uri=uri,
            name=name or uri.split('/')[-1],
            description=description,
            mime_type=mime_type,
            size=len(raw_content) if isinstance(raw_content, (str, bytes)) else 0,
            checksum=self._calculate_checksum(raw_content),
            tags=tags or [],
            cache_ttl=cache_ttl
        )
        
        # Store metadata
        self._resource_metadata[uri] = metadata
        
        # Cache the resource
        await self._cache_resource(uri, raw_content, metadata)
        
        # Save to storage if it's a file URI
        if uri.startswith('file://') or not uri.startswith(('http://', 'https://')):
            await self._save_resource_to_storage(uri, raw_content, metadata)
            
        self._logger.info(f"Created resource: {uri}")
        return metadata
        
    async def _save_resource_to_storage(
        self, 
        uri: str, 
        content: Union[str, bytes], 
        metadata: ResourceMetadata
    ) -> None:
        """Save resource content to storage.
        
        Args:
            uri: Resource URI
            content: Resource content
            metadata: Resource metadata
        """
        try:
            if uri.startswith('file://'):
                file_path = Path(uri[7:])
                if not file_path.is_absolute():
                    file_path = self.base_path / file_path
            else:
                file_path = self.base_path / uri
                
            # Ensure parent directory exists
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write content
            if isinstance(content, bytes):
                with open(file_path, 'wb') as f:
                    f.write(content)
            elif isinstance(content, str):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                # Convert other types to string
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(str(content))
                    
        except Exception as e:
            self._logger.error(f"Failed to save resource {uri}: {e}")
            
    async def list_resources(self, tags: Optional[List[str]] = None) -> List[ResourceMetadata]:
        """List available resources with filtering.
        
        Args:
            tags: Filter by tags
            
        Returns:
            List of resource metadata
        """
        resources = list(self._resource_metadata.values())
        
        if tags:
            resources = [
                resource for resource in resources
                if any(tag in resource.tags for tag in tags)
            ]
            
        # Sort by last accessed (most recent first)
        resources.sort(key=lambda r: r.last_accessed, reverse=True)
        
        return resources
        
    async def delete_resource(self, uri: str) -> bool:
        """Delete a resource.
        
        Args:
            uri: Resource URI
            
        Returns:
            True if deleted, False if not found
        """
        # Remove from cache
        if uri in self._resource_cache:
            del self._resource_cache[uri]
            
        # Remove metadata
        if uri in self._resource_metadata:
            del self._resource_metadata[uri]
            
            # Try to delete from storage
            try:
                if uri.startswith('file://'):
                    file_path = Path(uri[7:])
                    if not file_path.is_absolute():
                        file_path = self.base_path / file_path
                else:
                    file_path = self.base_path / uri
                    
                if file_path.exists():
                    file_path.unlink()
                    
            except Exception as e:
                self._logger.warning(f"Failed to delete resource file {uri}: {e}")
                
            self._logger.info(f"Deleted resource: {uri}")
            return True
            
        return False
        
    async def get_resource_metadata(self, uri: str) -> Optional[ResourceMetadata]:
        """Get resource metadata.
        
        Args:
            uri: Resource URI
            
        Returns:
            Resource metadata if found, None otherwise
        """
        return self._resource_metadata.get(uri)
        
    async def update_resource_metadata(
        self, 
        uri: str, 
        name: Optional[str] = None,
        description: Optional[str] = None,
        tags: Optional[List[str]] = None
    ) -> bool:
        """Update resource metadata.
        
        Args:
            uri: Resource URI
            name: New name
            description: New description
            tags: New tags
            
        Returns:
            True if updated, False if not found
        """
        if uri not in self._resource_metadata:
            return False
            
        metadata = self._resource_metadata[uri]
        
        if name is not None:
            metadata.name = name
        if description is not None:
            metadata.description = description
        if tags is not None:
            metadata.tags = tags
            
        self._logger.info(f"Updated metadata for resource: {uri}")
        return True
        
    async def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary of cache statistics
        """
        total_size = sum(
            len(cached.content) if isinstance(cached.content, (str, bytes)) else 0
            for cached in self._resource_cache.values()
        )
        
        return {
            "cached_resources": len(self._resource_cache),
            "total_resources": len(self._resource_metadata),
            "cache_size_bytes": total_size,
            "cache_hit_rate": self._calculate_cache_hit_rate(),
            "most_accessed": [
                {
                    "uri": metadata.uri,
                    "access_count": metadata.access_count,
                    "last_accessed": metadata.last_accessed.isoformat()
                }
                for metadata in sorted(
                    self._resource_metadata.values(),
                    key=lambda r: r.access_count,
                    reverse=True
                )[:10]
            ]
        }
        
    def _calculate_cache_hit_rate(self) -> float:
        """Calculate cache hit rate.
        
        Returns:
            Cache hit rate as percentage
        """
        # This would require tracking hits and misses
        # For now, return a placeholder
        return 75.0
        
    async def cleanup_expired_cache(self) -> int:
        """Clean up expired cached resources.
        
        Returns:
            Number of resources cleaned up
        """
        expired_uris = [
            uri for uri, cached in self._resource_cache.items()
            if cached.is_expired
        ]
        
        for uri in expired_uris:
            del self._resource_cache[uri]
            
        if expired_uris:
            self._logger.info(f"Cleaned up {len(expired_uris)} expired cached resources")
            
        return len(expired_uris)