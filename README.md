# 🐇 Similharity - Server Plugin for VectHare

> *The engine room behind VectHare.* Similharity is the SillyTavern server plugin that stores, searches and safely manages VectHare's vector collections across Standard (Vectra), LanceDB, Qdrant and Milvus.

![Version](https://img.shields.io/badge/version-4.0.0-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![Status](https://img.shields.io/badge/status-Active-brightgreen)

---

This is the plugin half of the VectHare fork. Install it together with the **VectHare** extension, using the **same version number** for both.

## 🎯 What is Similharity?

VectHare runs in your browser; Similharity runs inside the SillyTavern server. VectHare decides *what* to remember and *when* to recall it. Similharity does the server-side work: it talks to the vector databases, creates embeddings with SillyTavern's own embedding providers, and keeps track of which collection belongs to whom so nothing is changed or deleted by mistake.

### Why It's Needed

- 🗄️ Browsers can't talk to LanceDB, Qdrant or Milvus directly
- 🏷️ SillyTavern's built-in vector storage keeps little metadata per chunk
- 🔒 Deleting or moving collections safely needs checks on the server
- 👥 Multi-user servers need each user's data kept apart

---

## ✨ Key Features

### 📦 Four Vector Backends
- **Standard (Vectra)**: file-based, inside your SillyTavern data folder, with full chunk metadata
- **LanceDB**: disk-based, handles very large collections, installed with the plugin
- **Qdrant**: your own Qdrant server, local or cloud
- **Milvus**: your own Milvus server

### 🧠 Server-Side Embeddings
- Uses SillyTavern's own embedding providers (Transformers, OpenAI, Ollama, llama.cpp, vLLM, Cohere, Google and more)
- Nothing extra to configure: VectHare passes the provider you chose in its settings

### 🛡️ Safe Collection Management
- Every collection records what created it (backend, embedding source and model)
- Deletes, purges and moves are checked against that record before anything is removed
- Anything that can't be verified is kept, not guessed at

### 🗂️ VectHare's Own Store
- VectHare's collection list, settings per collection and locks live in one file per user, managed by the plugin
- Database keys (Qdrant/Milvus) are kept in a separate per-user key store that is never sent back to the browser

### 🤝 Tab and Device Coordination
- Two tabs or devices working on the same chat don't step on each other: the second one is told another tab is working and waits

### 👥 Multi-User Servers
- Each user's collections are kept apart
- Shared database connections are set up by the server admin
- Embedding requests to private-network addresses are blocked unless the admin allows them (see below)

---

## 🔗 Requirements

### Required
- **SillyTavern** (latest version) with server plugins enabled
- **Node.js 18** or newer (what SillyTavern already needs)
- **VectHare extension** with the **same version number**

### Optional (depending on the backend you use)
- **Qdrant** server, version **1.15 or newer**
- **Milvus** server, version **2.2.9 or newer**
- LanceDB needs nothing extra: it is installed with the plugin

---

## 💾 Installation

### Step 1: Download the Plugin

```bash
cd SillyTavern/plugins
git clone https://github.com/Samrwk99/Similharity-VectHare-Plugin.git similharity
cd similharity
npm install
```

> 💡 The plugin must sit directly inside `SillyTavern/plugins/` (e.g. `SillyTavern/plugins/similharity`): it uses SillyTavern's own embedding code from there.

### Step 2: Enable Server Plugins

Add to `config.yaml`:
```yaml
enableServerPlugins: true
```

### Step 3: Restart SillyTavern

The server console should show:
```
[similharity] Initializing v4.0.0...
```

### Step 4: Check It in VectHare

Open VectHare's **health panel**. It shows both versions and says whether VectHare and the plugin work together.

---

## 🔄 Updating

```bash
cd SillyTavern/plugins/similharity
git pull
npm install
```

Then restart SillyTavern and reload the page in your browser. Update VectHare to the same version at the same time.

---

## 🔢 Versions

VectHare and Similharity are released with the **same version number** - always install matching versions.

Behind the scenes, the plugin also reports a compatibility number to VectHare. If the two don't match, VectHare shows a warning when SillyTavern loads and in its health panel, saying which one to update. Actions that the other half can't support are refused with a message; stored data is not changed.

---

## 📁 Where Your Data Lives

| What | Where |
|------|-------|
| **Standard (Vectra)** collections | Your SillyTavern user folder, under `vectors/` |
| **LanceDB** collections | Your SillyTavern user folder, under `vectors/lancedb/` |
| **Qdrant / Milvus** collections | On your Qdrant or Milvus server |
| VectHare's collection list | `vectors/vecthare-owned-state/collections.json` in your user folder |
| Qdrant/Milvus keys | `secrets.json` beside it (never sent to the browser) |

> 💡 **Back up** your SillyTavern user folder as usual; VectHare's **Export** in the Database Browser also makes portable backups of single collections.

---

## 👥 Multi-User Servers

The plugin treats a SillyTavern server with more than one user as **multi-user** and adds a few rules there:

| Rule | Why |
|------|-----|
| **One Qdrant/Milvus connection** for everyone, set up by an **admin** | Users can't point the shared server elsewhere; non-admins use the admin's connection |
| **No loopback/private-network URLs** for embedding requests | Stops users reaching other machines on the server's network |
| **Collections are per user** | Users can't see, change or delete each other's data |

To allow private-network embedding servers (for example a local Ollama or llama.cpp on the same machine) on a multi-user server, the admin can start SillyTavern with:
```bash
SIMILHARITY_ALLOW_PRIVATE_URLS=1
```

Single-user setups have none of these limits.

---

## 🐛 Troubleshooting

### The plugin doesn't load
1. Check `enableServerPlugins: true` in `config.yaml`
2. Check the folder is directly inside `SillyTavern/plugins/`
3. Run `npm install` in the plugin folder and look for errors
4. Restart SillyTavern and look for `[similharity]` lines in the server console

### "Plugin and extension do not match"
1. Install the same version of VectHare and Similharity
2. Restart SillyTavern
3. Reload the page

### "Only an admin can set up the Qdrant/Milvus connection"
1. On a multi-user server, an admin selects Qdrant/Milvus in VectHare and connects once
2. Other users then use that connection automatically

### "Blocked: … loopback/private-network URLs are not allowed"
1. Use a public embedding endpoint, or
2. Ask the admin to set `SIMILHARITY_ALLOW_PRIVATE_URLS=1`

### Qdrant or Milvus refuses to store collections
1. **Qdrant**: update the server to 1.15 or newer
2. **Milvus**: update the server to 2.2.9 or newer
3. Check the server is running and reachable from the SillyTavern machine

### Something was "kept" during a delete or purge
The plugin keeps anything it can't verify as belonging to that collection. Run VectHare's **Diagnostics** and check the **Database Browser**; the message says what couldn't be verified.

---

## 🧩 For Developers

All routes live under `/api/plugins/similharity/`. `GET /api/plugins/similharity/health` returns the plugin's version, its compatibility number (`protocol`), the supported backends and its capabilities. VectHare reads this once per page load.

---

## 📜 License

MIT License - See LICENSE file for details.

---

## 🙏 Credits

**VectHare** and **Similharity** originally created with 💜 by **Coneja Chibi**

Special thanks to the SillyTavern community for feedback and testing!

---

*"Your memories, safely stored."* 🐇✨
