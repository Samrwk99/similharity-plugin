/** Server-side embedding-provider adapter. This is an API gateway for the configured user endpoint,
 * not a general-purpose tokenizer. Never silently truncate or guess exact token counts. */
const DEFAULTS = {
    nanogpt: 'https://nano-gpt.com/api/v1',
    openrouter: 'https://openrouter.ai/api/v1',
    koboldcpp: 'http://127.0.0.1:5001',
    vllm: 'http://127.0.0.1:8000',
    bananabread: 'http://127.0.0.1:8008',
};
export function providerBase(source, apiUrl) {
    const raw = (apiUrl || DEFAULTS[source] || '').trim();
    if (!raw) throw new Error(`Explicit ${source} embedding endpoint is required`);
    const url = new URL(raw);
    if (!['http:', 'https:'].includes(url.protocol)) throw new Error('Only HTTP(S) embedding endpoints are supported');
    if (url.username || url.password) throw new Error('Credentials in embedding URLs are not permitted');
    return url;
}
export function providerRoute(source, apiUrl, route, customEndpointMode='base') {
    const url = providerBase(source, apiUrl);
    if (source === 'custom') {
        if (!['base','exact'].includes(customEndpointMode)) throw new Error('Unknown Custom API endpoint URL mode');
        if (customEndpointMode === 'exact') {
            if (route !== 'embeddings') throw new Error('Exact Custom API only defines an embeddings endpoint; auxiliary routes are unavailable');
            if (url.hash) throw new Error('Custom API Exact Endpoint URL cannot contain a fragment (#)');
            // Preserve the exact path AND query string, including signed URL parameters.
            return url;
        }
    }
    // OpenAI-compatible base URLs may end with /v1, /api/v1 or /api/v1/embeddings.
    let path = url.pathname.replace(/\/+$/, '').replace(/\/embeddings$/, '');
    if (!path.endsWith('/v1') && !path.endsWith('/api/v1')) path += '/v1';
    url.pathname = `${path}/${route.replace(/^\/+/, '')}`.replace(/\/+/g, '/');
    url.search = ''; url.hash = '';
    return url;
}
export async function openAIEmbeddings(source, texts, model, apiUrl, apiKey, fetcher=fetch, customEndpointMode='base') {
    if (!model || !String(model).trim()) throw new Error(`Select an embedding model for ${source}`);
    const headers = {'Content-Type':'application/json'};
    if (apiKey) headers.Authorization = `Bearer ${apiKey}`;
    const response = await fetcher(providerRoute(source,apiUrl,'embeddings',customEndpointMode), {
        method:'POST',headers,
        body:JSON.stringify({input:texts,model}),signal:AbortSignal.timeout(60000),
    });
    if (!response.ok) throw new Error(`${source} embeddings HTTP ${response.status}: ${(await response.text()).slice(0,400)}`);
    const data = await response.json();
    const rows = data?.data;
    if (!Array.isArray(rows) || rows.length!==texts.length) throw new Error(`${source}: incomplete embeddings response`);
    const sorted=[...rows].sort((a,b)=>a.index-b.index);
    if (!sorted.every((row,index)=>row.index===index && Array.isArray(row.embedding) && row.embedding.length>0 && row.embedding.every(Number.isFinite))) throw new Error(`${source}: invalid or missing embedding vector`);
    return sorted.map(row=>row.embedding);
}
export async function strictOllamaEmbedding(text, apiUrl, model, keep=false, apiKey='', fetcher=fetch) {
    if (!model) throw new Error('Ollama embedding model is required');
    const base=providerBase('ollama',apiUrl);
    const url=new URL('/api/embed',base);
    const headers={'Content-Type':'application/json'};
    if (apiKey) headers.Authorization=`Bearer ${apiKey}`;
    const response=await fetcher(url,{method:'POST',headers,body:JSON.stringify({model,input:text,truncate:false,keep_alive:keep?-1:undefined}),signal:AbortSignal.timeout(60000)});
    if(!response.ok) throw new Error(`Ollama embedding HTTP ${response.status} (truncate:false; text was NOT silently shortened): ${(await response.text()).slice(0,400)}`);
    const data=await response.json();
    const vector=data?.embeddings?.[0];
    if(!Array.isArray(vector)||!vector.length||!vector.every(Number.isFinite)) throw new Error('Ollama returned no valid embedding');
    return vector;
}
export async function providerCapabilities(source,apiUrl,model,apiKey='',fetcher=fetch,customEndpointMode='base') {
    const base=providerBase(source,apiUrl);
    if(source==='custom' && customEndpointMode==='exact') {
        providerRoute(source,apiUrl,'embeddings',customEndpointMode); // Validate the exact URL.
        // No implicit /models or /tokenize route exists for an arbitrary endpoint.
        return {success:true,modelId:model||null,limits:null,tokenizer:'unverified',endpointModeSupported:true,endpointMode:'exact'};
    }
    if(source==='koboldcpp') return {success:true,modelId:model||null,limits:null,tokenizer:'unverified',reason:'Embedding model context (--embeddingsmaxctx) is independent from the text model and is not safely discoverable from /v1/models'};
    const headers={}; if(apiKey)headers.Authorization=`Bearer ${apiKey}`;
    const paths=source==='openrouter'?['embeddings/models']:['models'];
    for (const path of paths){
        try {
            const res=await fetcher(providerRoute(source,apiUrl,path),{headers,signal:AbortSignal.timeout(7000)});
            if(!res.ok)continue;
            const data=await res.json();
            const row=data?.data?.find(x=>x.id===model||x.name===model) || (!model && source!=='koboldcpp' && data?.data?.length===1 ? data.data[0] : null);
            if(!row)continue;
            const raw=row.context_length??row.top_provider?.context_length??row.max_model_len??row.max_input_tokens??row.input_token_limit;
            const maximum=Number(raw);
            return {success:true,modelId:row.id||model,limits:Number.isSafeInteger(maximum)&&maximum>0?{reportedContextTokens:maximum}:null,tokenizer:'unverified'};
        }catch{} // Discovery is optional; never silently claim to have learned a limit.
    }
    return {success:true,modelId:model||null,limits:null,tokenizer:'unverified',...(source==='custom'?{endpointModeSupported:true,endpointMode:'base'}:{})};
}
