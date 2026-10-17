import assert from 'node:assert/strict';
import {providerRoute, openAIEmbeddings, providerCapabilities} from '../embedding-providers.js';
let passed=0;
async function check(name, fn){await fn();passed++;console.log('PASS',name);}
await check('Existing Base mode appends /v1/embeddings to a hostname', () => {
  assert.equal(providerRoute('custom','http://127.0.0.1:9712','embeddings').href,'http://127.0.0.1:9712/v1/embeddings');
});
await check('Existing Base mode does not duplicate /v1/embeddings', () => {
  assert.equal(providerRoute('custom','https://example.com/proxy/v1/embeddings','embeddings').href,'https://example.com/proxy/v1/embeddings');
});
await check('Exact mode preserves arbitrary path, query and port', () => {
  const supplied='http://127.0.0.1:9977/weird/embedding-handler?token=a%2Fb&scope=1';
  assert.equal(providerRoute('custom',supplied,'embeddings','exact').href,supplied);
});
await check('Exact mode never attempts to append a version or embedding path', () => {
  assert.equal(providerRoute('custom','https://api.example.com/model/embed','embeddings','exact').href,'https://api.example.com/model/embed');
});
await check('Exact mode rejects unsupported auxiliary routes', () => {
  assert.throws(() => providerRoute('custom','https://api.example.com/embed','models','exact'),/only defines an embeddings endpoint/);
});
await check('Exact mode rejects URL credentials, fragments and non-HTTP URLs', () => {
  assert.throws(() => providerRoute('custom','http://user:pass@example.com/embed','embeddings','exact'),/credentials/i);
  assert.throws(() => providerRoute('custom','http://example.com/embed#discarded','embeddings','exact'),/fragment/i);
  assert.throws(() => providerRoute('custom','file:///etc/passwd','embeddings','exact'),/HTTP\(S\)/i);
});
await check('Exact mode delivers OpenAI-compatible request to exact URL, with credentials and correct vector order', async() => {
  let requested;
  const embeddings=await openAIEmbeddings('custom',['text A','text B'],'my-embedding-model',
    'https://host.invalid/path/to/embed?sig=xyz','api-secret',async(url,opts) => {
      requested={url:url.href,method:opts.method,headers:opts.headers,body:JSON.parse(opts.body)};
      return {ok:true,json:async()=>({data:[{index:1,embedding:[0,2]},{index:0,embedding:[2,0]}]})};
    },'exact');
  assert.equal(requested.url,'https://host.invalid/path/to/embed?sig=xyz');
  assert.equal(requested.method,'POST');
  assert.equal(requested.headers.Authorization,'Bearer api-secret');
  assert.deepEqual(requested.body,{input:['text A','text B'],model:'my-embedding-model'});
  assert.deepEqual(embeddings,[[2,0],[0,2]]);
});
await check('Exact mode validates embedding responses; a malformed result fails', async() => {
  await assert.rejects(openAIEmbeddings('custom',['text'],'m','https://host.invalid/custom','',
    async()=>({ok:true,json:async()=>({data:[{index:0,embedding:[]}]})}),'exact'),/invalid or missing embedding/);
});
await check('Exact mode capability check advertises support and does not probe arbitrary URL', async() => {
  const result=await providerCapabilities('custom','https://host.invalid/special/embed','model-id','',
    async()=>{throw Error('Network probe forbidden for exact URL');},'exact');
  assert.equal(result.endpointMode,'exact');
  assert.equal(result.endpointModeSupported,true);
  assert.equal(result.limits,null);
  assert.equal(result.modelId,'model-id');
});
await check('NanoGPT route is unchanged by Custom API mode', () => {
  assert.equal(providerRoute('nanogpt','https://nano-gpt.com/api/v1','embeddings','exact').href,'https://nano-gpt.com/api/v1/embeddings');
});
console.log(`\n${passed} exact endpoint transport checks passed`);
