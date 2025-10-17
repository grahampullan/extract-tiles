// server.mjs
import Fastify from 'fastify';
import cors from '@fastify/cors';
import compress from '@fastify/compress';
import fastifyStatic from '@fastify/static';
import path from 'node:path';
import fs from 'node:fs/promises';
import { createReadStream } from 'node:fs';
import { fileURLToPath } from 'node:url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
const __ROOT = process.cwd();
const PUBLIC_ROOT = path.join(__ROOT, 'public');
const DATA_ROOT = path.join(__ROOT, 'tiles_out');

const app = Fastify({ logger: true });

await app.register(cors, { origin: true });
await app.register(compress, { global: true });

// Viewer
await app.register(fastifyStatic, {
  root: PUBLIC_ROOT,
  prefix: '/',
  wildcard: true,
  index: ['index.html']
});

// Tiles (supports all layouts: UV-quadtree, world-octree, atlas-quadtree)
await app.register(fastifyStatic, {
  root: DATA_ROOT,
  prefix: '/tiles/',
  decorateReply: false,
  setHeaders(res, filePath) {
    if (filePath.endsWith('.glb')) {
      res.setHeader('Content-Type', 'model/gltf-binary');
    } else {
      res.setHeader('Content-Type', 'application/octet-stream');
    }
    res.setHeader('Cache-Control', 'public, max-age=86400, immutable');
  }
});

// Manifest
app.get('/manifest/:extract/:time.json', async (req, reply) => {
  const { extract, time } = req.params;
  const p = path.join(DATA_ROOT, extract, `manifest_${time}.json`);
  try {
    await fs.access(p);
    reply.header('Content-Type', 'application/json');
    reply.header('Cache-Control', 'public, max-age=3600');
    return reply.send(createReadStream(p));
  } catch {
    return reply.code(404).send({ error: 'manifest not found' });
  }
});

// API to list available extracts
app.get('/api/extracts', async (req, reply) => {
  try {
    const dirs = await fs.readdir(DATA_ROOT);
    const extracts = [];

    for (const dir of dirs) {
      const dirPath = path.join(DATA_ROOT, dir);
      const stat = await fs.stat(dirPath);

      if (stat.isDirectory()) {
        // Look for manifest files
        const files = await fs.readdir(dirPath);
        const manifests = files
          .filter(f => f.startsWith('manifest_') && f.endsWith('.json'))
          .map(f => {
            const match = f.match(/manifest_(\d+)\.json/);
            return match ? parseInt(match[1]) : null;
          })
          .filter(t => t !== null);

        if (manifests.length > 0) {
          extracts.push({
            name: dir,
            times: manifests.sort((a, b) => a - b)
          });
        }
      }
    }

    return reply.send(extracts);
  } catch (err) {
    app.log.error(err);
    return reply.code(500).send({ error: 'Failed to list extracts' });
  }
});

const PORT = process.env.PORT || 8080;
await app.listen({ port: Number(PORT), host: '0.0.0.0' });
console.log(`Server running on http://localhost:${PORT}`);
