/** Dependency-free provider proxy checks. */
import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';
const source=fs.readFileSync(new URL('../index.js',import.meta.url),'utf8');
const begin=source.indexOf("    router.post('/embedding/capabilities'");
const end=source.indexOf('    /**\n     * GET /api/plugins/similharity/sources',begin);
assert.ok(begin>0 && end>begin);
const routes=new Map();const calls=[];
let ollamaTokenizer=true;
const ctx={URL,AbortSignal,console,formatError:e=>e.message,
  router:{post:(route,fn)=>routes.set(route,fn)},
  fetch:async(url,options={})=>{
    const name=new URL(url).pathname;const body=options.body?JSON.parse(options.body):{};
    calls.push({name,body});
    if(name==='/api/show')return {ok:true,json:async()=>({model_info:{'nomic.context_length':8192}})};
    if(name==='/api/tags')return {ok:true,json:async()=>({models:[{name:'nomic-embed-text:latest',digest:'digest-abc'}]})};
    if(name==='/v1/models')return {ok:true,json:async()=>({data:[{id:'gguf-embed'}]})};
    if(name==='/props')return {ok:true,json:async()=>({n_ubatch:1024,n_ctx:4096,model_path:'/models/embed.gguf'})};
    if(name==='/api/tokenize' && !ollamaTokenizer)return {ok:false,status:404};
    if(name==='/api/tokenize'||name==='/tokenize')return {ok:true,json:async()=>({tokens:[1,2,3]})};
    throw Error(`Unknown route ${name}`);
  },
};
vm.runInNewContext(source.slice(begin,end),ctx);
async function request(route,body){
  let status=200,data;
  const res={status(n){status=n;return this;},json(obj){data=obj;return this;}};
  await routes.get(route)({body},res);return {status,data};
}
let checks=0;const ok=name=>{checks++;console.log('PASS',name)};
let r=await request('/embedding/capabilities',{source:'ollama',apiUrl:'http://localhost:11434',model:'nomic-embed-text:latest'});
assert.equal(r.status,200);assert.equal(r.data.modelId,'nomic-embed-text:latest');assert.equal(r.data.modelPath,'digest-abc');
assert.equal(r.data.limits.reportedContextTokens,8192);assert.equal(r.data.limits.maxTokens,undefined);
ok('Ollama /api/show provides architectural context, not falsely verified embedding budget');
r=await request('/embedding/capabilities',{source:'llamacpp',apiUrl:'http://localhost:9977'});
assert.equal(r.data.modelId,'gguf-embed');assert.equal(r.data.limits.maxTokens,1024);
ok('llama.cpp uses the configured endpoint and physical batch limit');
r=await request('/embedding/tokenize',{source:'ollama',apiUrl:'http://localhost:11434',model:'nomic-embed-text:latest',text:'hello'});
assert.equal(r.data.count,3);assert.equal(calls.at(-1).body.model,'nomic-embed-text:latest');
ok('Ollama tokenizer uses the selected embedding model');
ollamaTokenizer=false;
r=await request('/embedding/tokenize',{source:'ollama',apiUrl:'http://localhost:11434',model:'nomic-embed-text:latest',text:'hello'});
assert.equal(r.status,422);assert.match(r.data.error,/not expose model-accurate/);
ok('Ollama missing tokenizer fails explicitly, does not estimate');
r=await request('/embedding/capabilities',{source:'ollama',apiUrl:'http://localhost:11434'});
assert.equal(r.data.modelId,null);
ok('Ollama capabilities refuse ambiguous model-less identification');
console.log(`\n${checks} P4 plugin provider contract checks passed`);
