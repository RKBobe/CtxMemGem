import 'dotenv/config';
import express from 'express';
import axios from 'axios';
import { ChromaClient } from 'chromadb';
import { pipeline } from '@xenova/transformers';
import { GoogleGenerativeAI } from '@google/generative-ai';

// --- Global Variables and Configuration ---
const PORT = 3000;
const CLIENT_ID = process.env.GITHUB_CLIENT_ID;
const CLIENT_SECRET = process.env.GITHUB_CLIENT_SECRET;
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
let accessToken = null;

// --- Embedding Model Singleton ---
class EmbeddingPipeline {
    static task = 'feature-extraction';
    static model = 'Xenova/all-MiniLM-L6-v2';
    static instance = null;

    static async getInstance(progress_callback = null) {
        if (this.instance === null) {
            this.instance = await pipeline(this.task, this.model, { quantized: true, progress_callback });
        }
        return this.instance;
    }
}

class MyDummyEmbeddingFunction {
    constructor() {}
    async generate(texts) { return texts.map(() => []); }
}
const dummyEmbedder = new MyDummyEmbeddingFunction();

// --- Main Server Logic ---
async function startServer() {
    const genAI = new GoogleGenerativeAI(GEMINI_API_KEY);
    const llm_model = genAI.getGenerativeModel({ model: "gemini-1.5-flash"});
    console.log('Google AI client initialized.');

    const chroma = new ChromaClient({ path: 'http://localhost:8000' });
    console.log('ChromaDB client initialized.');

    console.log('Initializing AI embedding pipeline...');
    const extractor = await EmbeddingPipeline.getInstance(console.log);
    console.log('AI embedding pipeline loaded successfully.');
    
    const app = express();
    app.use(express.json());

    // --- API Routes ---
    app.get('/auth/github', (req, res) => {
        const codespaceName = process.env.CODESPACE_NAME;
        const publicUrlBase = codespaceName ? `https://${codespaceName}-${PORT}.app.github.dev` : `http://localhost:${PORT}`;
        const redirect_uri = `${publicUrlBase}/auth/github/callback`;
        const url = `https://github.com/login/oauth/authorize?client_id=${CLIENT_ID}&redirect_uri=${redirect_uri}&scope=repo`;
        res.redirect(url);
    });

    app.get('/auth/github/callback', async (req, res) => {
        const { code } = req.query;
        try {
            const response = await axios.post('https://github.com/login/oauth/access_token', {
                client_id: CLIENT_ID, client_secret: CLIENT_SECRET, code: code,
            }, { headers: { 'Accept': 'application/json' } });
            accessToken = response.data.access_token;
            res.send('Authentication successful!');
        } catch (error) {
            console.error('Error obtaining access token:', error.message);
            res.status(500).send('Authentication Failed.');
        }
    });

    app.get('/api/repos/:owner/:repo/process', async (req, res) => {
        if (!accessToken) return res.status(401).send('Not authenticated.');
        const { owner, repo } = req.params;
        const collectionName = `${owner}-${repo}`.replace(/[^a-zA-Z0-9\-\_]/g, '_');
        
        try {
            await chroma.deleteCollection({ name: collectionName }).catch(() => {});
            const collection = await chroma.createCollection({ name: collectionName, embeddingFunction: dummyEmbedder });

            const treeResponse = await axios.get(`https://api.github.com/repos/${owner}/${repo}/git/trees/main?recursive=1`, { headers: { 'Authorization': `token ${accessToken}` } });
            const textFilePaths = treeResponse.data.tree.filter(f => f.type === 'blob' && !isBinary(f.path)).map(f => f.path);

            for (const filePath of textFilePaths) {
                try {
                    const fileUrl = `https://api.github.com/repos/${owner}/${repo}/contents/${encodeURIComponent(filePath)}`;
                    const fileRes = await axios.get(fileUrl, { headers: { 'Authorization': `token ${accessToken}` }});
                    if (!fileRes.data || !fileRes.data.content) continue;

                    const content = Buffer.from(fileRes.data.content, 'base64').toString('utf8');
                    const chunks = chunkText(content);
                    if (chunks.length === 0) continue;

                    const embeddingsTensor = await extractor(chunks, { pooling: 'mean', normalize: true });
                    const embeddings = [];
                    for (let i = 0; i < embeddingsTensor.dims[0]; ++i) {
                        embeddings.push(Array.from(embeddingsTensor.data.slice(i * embeddingsTensor.dims[1], (i + 1) * embeddingsTensor.dims[1])));
                    }
                    
                    await collection.add({
                        ids: chunks.map((_, i) => `${filePath}-${i}`),
                        embeddings: embeddings,
                        documents: chunks
                    });
                    console.log(` -> Stored ${chunks.length} chunks from ${filePath}`);
                } catch (loopError) {
                    console.error(`[ERROR] Skipping file '${filePath}':`, loopError.message);
                }
            }
            res.json({ message: `Processing complete.` });
        } catch (error) {
            console.error(`A critical error occurred:`, error.message);
            res.status(500).send(`Failed to process repository.`);
        }
    });
    
    app.post('/api/ask/:owner/:repo', async (req, res) => {
        const { owner, repo } = req.params;
        const { query } = req.body;
        const collectionName = `${owner}-${repo}`.replace(/[^a-zA-Z0-9\-\_]/g, '_');
        console.log(`\n--- Received query for '${collectionName}': "${query}" ---`);

        try {
            console.log('[ASK_DEBUG] Step 1: Getting collection...');
            const collection = await chroma.getCollection({ name: collectionName, embeddingFunction: dummyEmbedder });
            
            console.log('[ASK_DEBUG] Step 2: Generating query embedding...');
            const queryEmbedding = await extractor(query, { pooling: 'mean', normalize: true });
            
            console.log('[ASK_DEBUG] Step 3: Querying ChromaDB...');
            const results = await collection.query({
                queryEmbeddings: [Array.from(queryEmbedding.data)], nResults: 5
            });
            console.log('[ASK_DEBUG] Step 4: Building prompt...');
            const context = results.documents[0].join('\n---\n');
            const augmentedPrompt = `Based on the following context from my codebase, answer the question.\nContext:\n${context}\n\nQuestion: ${query}`;
            
            console.log('[ASK_DEBUG] Step 5: Sending prompt to Gemini...');
            const result = await llm_model.generateContent(augmentedPrompt);
            
            console.log('[ASK_DEBUG] Step 6: Extracting text from Gemini response...');
            const response = result.response;
            const text = response.text();

            console.log('[ASK_DEBUG] Step 7: Sending final answer to client.');
            res.json({ answer: text });
        } catch (error) {
            console.error(`[ASK_ERROR] Error during synthesis:`, error);
            res.status(500).send('Failed to complete synthesis.');
        }
    });

    app.listen(PORT, () => console.log(`\nServer started on http://localhost:${PORT}`));
}

// --- Helper Functions ---
function chunkText(text, chunkSize = 200, overlap = 50) {
    const words = text.split(/\s+/);
    if (words.length === 0) return [];
    const chunks = [];
    let currentChunk = [];
    for (const word of words) {
        currentChunk.push(word);
        if (currentChunk.length >= chunkSize) {
            chunks.push(currentChunk.join(' '));
            currentChunk = currentChunk.slice(currentChunk.length - overlap);
        }
    }
    if (currentChunk.length > 0) chunks.push(currentChunk.join(' '));
    return chunks;
}
function isBinary(filePath) { return false; }

// --- Execute the server startup ---
startServer().catch(err => {
    console.error('Failed to start server:', err);
    process.exit(1);
});