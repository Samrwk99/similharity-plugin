/** Run the actual plugin profile routes and incremental journal against disk. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import syncFs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import vm from 'node:vm';
import {createProfileJournal} from '../profile-journal.js';
const full=syncFs.readFileSync(new URL('../index.js',import.meta.url),'utf8');
const from=full.indexOf('const profileFileLocks=new Map();');
const to=full.indexOf('    /**\n     * Proxy metadata',from);
assert.ok(from>0&&to>from);
const dir=await fs.mkdtemp(path.join(os.tmpdir(),'vecthare-journal-'));
const target=path.join(dir,'manifest.json');
const routes=new Map();
const sandbox={router:{post:(name,handler)=>routes.set(name,handler)},profilePath:()=>target,fs,
 formatError:e=>String(e.message),path,Buffer,Map,JSON,Date,Math,process,createProfileJournal,console};
vm.runInNewContext(full.slice(from,to),sandbox);
function call(route,body){return new Promise(async resolve=>{const res={code:200,status(code){this.code=code;return this;},json(payload){resolve({code:this.code,payload});}};await routes.get(route)({body},res);});}
const key='qdrant:llamacpp:chatA';
const before={key,profile:{schemaVersion:1,embedding:{model:'A'}},manifest:{schemaVersion:1,chatUUID:'uuid-1',entries:[{hash:1,text:'first'}],revision:0,updatedAt:'old'}};
let passes=0;function ok(name){passes++;console.log('PASS',name)}
try {
 let r=await call('/collection/profile/set',before);assert.equal(r.code,200);
 const original=await fs.readFile(target,'utf8');
 const patch={key,chatUUID:'uuid-1',upserts:[{hash:2,text:'second',metadata:{messageIds:[4]}}],removedHashes:[],expectedRevision:0,revision:1,expectedUpdatedAt:'old',updatedAt:'new'};
 r=await call('/collection/profile/manifest/patch',patch);
 assert.equal(r.code,200);assert.equal(r.payload.entries,2);
 assert.equal(await fs.readFile(target,'utf8'),original,'small patch must NOT rewrite the complete snapshot');
 assert.match(await fs.readFile(`${target}.journal`,'utf8'),/"revision":1/);
 let read=await call('/collection/profile/get',{key});assert.equal(read.code,200);
 assert.equal(read.payload.manifest.entries.length,2);ok('incremental patch durably appends without rewriting full snapshot');
 r=await call('/collection/profile/manifest/patch',patch);assert.equal(r.code,409);ok('stale revision cannot overwrite committed manifest');
 r=await call('/collection/profile/manifest/patch',{...patch,expectedRevision:1,revision:2,expectedUpdatedAt:'new',updatedAt:'next',chatUUID:'wrong'});
 assert.equal(r.code,409);ok('wrong chat is rejected');
 r=await call('/collection/profile/manifest/patch',{...patch,expectedRevision:1,revision:2,expectedUpdatedAt:'new',updatedAt:'next',upserts:[],removedHashes:[1]});
 assert.equal(r.code,200);
 read=await call('/collection/profile/get',{key});assert.deepEqual(Array.from(read.payload.manifest.entries.map(x=>x.hash)),[2]);ok('remove-only patch preserves unchanged entries');
 const concurrentBase={...patch,expectedRevision:2,revision:3,expectedUpdatedAt:'next',updatedAt:'parallel'};
 const [a,b]=await Promise.all([
  call('/collection/profile/manifest/patch',{...concurrentBase,upserts:[{hash:3,text:'third'}],removedHashes:[]}),
  call('/collection/profile/manifest/patch',{...concurrentBase,upserts:[{hash:4,text:'fourth'}],removedHashes:[]})
 ]);
 assert.deepEqual([a.code,b.code].sort((x,y)=>x-y),[200,409]);ok('concurrent patches cannot acknowledge same revision');
 let reloaded=createProfileJournal(fs,async(p,v)=>fs.writeFile(p,v));
 const disk=await reloaded.load(target);assert.equal(disk.state.manifest.revision,3);assert.equal(disk.state.manifest.entries.length,2);
 ok('a fresh process replays committed journal revisions correctly');
 r=await call('/collection/profile/set',{...before,manifest:{...before.manifest,revision:0}});
 assert.equal(r.code,409);ok('stale whole-manifest save cannot erase newer checkpoints');
 const next=disk.state;
 r=await call('/collection/profile/set',next);assert.equal(r.code,200);
 assert.equal(await fs.readFile(`${target}.journal`,'utf8').catch(e=>e.code==='ENOENT'?'absent':null),'absent');
 ok('full checkpoint compacts and removes journal');
 await fs.appendFile(`${target}.journal`,'{"revision":4');
 const corrupted=createProfileJournal(fs,async(p,v)=>fs.writeFile(p,v));
 await assert.rejects(corrupted.load(target),/partial write/);
 ok('truncated crash journal cannot be mistaken for a clean collection');
 const compactFile=path.join(dir,'compact.json');
 const compactJournal=createProfileJournal(fs,async(name,contents)=>fs.writeFile(name,contents),{compactEvery:3});
 const recipe={version:2,immutable:true};
 await compactJournal.set(compactFile,{...before,profile:{...before.profile,recipe}});
 for(let i=1;i<=3;i++)await compactJournal.patch(compactFile,{
   key,chatUUID:'uuid-1',upserts:[{hash:10+i,text:`part ${i}`}],removedHashes:[],
   expectedRevision:i-1,revision:i,expectedUpdatedAt:i===1?'old':`t${i-1}`,updatedAt:`t${i}`
 });
 const snapshot=JSON.parse(await fs.readFile(compactFile,'utf8'));
 assert.equal(snapshot.manifest.revision,3);
 assert.equal(snapshot.manifest.entries.length,4);
 const fresh=await createProfileJournal(fs,async(n,v)=>fs.writeFile(n,v)).load(compactFile);
 assert.equal(fresh.state.manifest.entries.length,4);
 ok('periodic journal compaction preserves every committed chunk and revision');
 await assert.rejects(compactJournal.set(compactFile,{...snapshot,
   profile:{...snapshot.profile,recipe:{version:2,immutable:false}}}),/immutable/);
 ok('server rejects in-place changes to collection embedding recipe');
 await assert.rejects(compactJournal.set(compactFile,{...snapshot,manifest:{...snapshot.manifest,entries:[]}}),/Same manifest revision/);
 ok('same revision with different contents cannot silently overwrite a manifest');
}finally{await fs.rm(dir,{recursive:true,force:true});}
console.log(`\n${passes} manifest journal checks passed`);
