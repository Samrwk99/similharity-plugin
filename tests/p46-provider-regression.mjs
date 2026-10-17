import assert from 'node:assert/strict';
import {openAIEmbeddings, strictOllamaEmbedding, providerRoute, providerCapabilities} from '../embedding-providers.js';
let passed=0;
async function test(name,fn){await fn();passed++;console.log('PASS',name);}
await test('NanoGPT URL builds exactly /api/v1/embeddings',()=>{
  assert.equal(providerRoute('nanogpt','https://nano-gpt.com/api/v1','embeddings').href,'https://nano-gpt.com/api/v1/embeddings');
});
await test('Custom v1 and embeddings URL normalize without duplicating v1',()=>{
  assert.equal(providerRoute('custom','http://127.0.0.1:9977/v1/embeddings','embeddings').href,'http://127.0.0.1:9977/v1/embeddings');
});
await test('Custom API requires an actual endpoint',()=>assert.throws(()=>providerRoute('custom','','embeddings'),/endpoint is required/i));
await test('Custom API disallows URL-embedded user/password',()=>assert.throws(()=>providerRoute('custom','http://u:p@localhost:9977','embeddings'),/credentials/i));
await test('OpenAI-compatible request sends correct model, bearer token and preserves ordering',async()=>{
  let captured;
  const result=await openAIEmbeddings('nanogpt',['one','two'],'embed-model','https://nano-gpt.com/api/v1','private-token',async(url,opts)=>{
    captured={url:String(url),opts};
    return {ok:true,json:async()=>({data:[{index:1,embedding:[0,1]},{index:0,embedding:[1,0]}]})};
  });
  assert.deepEqual(result,[[1,0],[0,1]]);
  assert.equal(captured.opts.headers.Authorization,'Bearer private-token');
  assert.deepEqual(JSON.parse(captured.opts.body),{input:['one','two'],model:'embed-model'});
});
await test('Incomplete OpenAI-compatible results cannot be marked successful',async()=>{
  await assert.rejects(openAIEmbeddings('custom',['one','two'],'m','http://localhost:9997/v1','',async()=>({ok:true,json:async()=>({data:[{index:0,embedding:[1]}]})})),/incomplete/i);
});
await test('Non-numeric vectors are rejected',async()=>{
  await assert.rejects(openAIEmbeddings('custom',['a'],'m','http://localhost:9997/v1','',async()=>({ok:true,json:async()=>({data:[{index:0,embedding:[NaN]}]})})),/invalid/i);
});
await test('Ollama always sets truncate:false on /api/embed',async()=>{
  let captured;
  const vector=await strictOllamaEmbedding('full text','http://127.0.0.1:11434','m',false,'',async(url,opts)=>{
    captured={url:String(url),opts};
    return {ok:true,json:async()=>({embeddings:[[1,2,3]]})};
  });
  assert.deepEqual(vector,[1,2,3]);
  assert.equal(captured.url,'http://127.0.0.1:11434/api/embed');
  assert.equal(JSON.parse(captured.opts.body).truncate,false);
});
await test('Ollama over-limit failure does not silently fall back to old /api/embeddings',async()=>{
  let calls=0;
  await assert.rejects(strictOllamaEmbedding('huge','http://localhost:11434','m',false,'',async()=>{calls++;return {ok:false,status:400,text:async()=> 'context length exceeded'};}),/NOT silently shortened/);
  assert.equal(calls,1);
});
await test('Metadata advertised limit is labelled context, not verified batch size',async()=>{
  const data=await providerCapabilities('openrouter','https://openrouter.ai/api/v1','embed','',async()=>({ok:true,json:async()=>({data:[{id:'embed',context_length:1024}]})}));
  assert.equal(data.limits.reportedContextTokens,1024);
  assert.equal(data.tokenizer,'unverified');
});
await test('KoboldCpp does not mistake text-model context for separate embeddings-model limit',async()=>{
  const result=await providerCapabilities('koboldcpp','http://127.0.0.1:5001','bge-embedding-model','',async()=>{throw Error('Text model endpoint must not be used');});
  assert.equal(result.limits,null);assert.equal(result.modelId,'bge-embedding-model');
});
console.log(`\n${passed} P46 provider transport checks passed`);
