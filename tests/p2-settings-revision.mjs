import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createSettingsRevisionStore} from '../settings-revision-store.js';
const home=await fs.mkdtemp(path.join(os.tmpdir(),'vecthare-settings-rev-'));
const store=createSettingsRevisionStore(fs);
try{
 const zero=await store.read(home);
 assert.equal(zero.revision,0);
 const a={collections:{chatA:{enabled:true}},top_k:0};
 let ack=await store.commit(home,0,a);assert.equal(ack.revision,1);
 let saved=await store.read(home);assert.deepEqual(saved.snapshot,a);
 await assert.rejects(store.commit(home,0,{top_k:5}),/stale/i);
 const all=await Promise.allSettled([store.commit(home,1,{top_k:1}),store.commit(home,1,{top_k:2})]);
 assert.equal(all.filter(x=>x.status==='fulfilled').length,1,'Only one writer can commit the revision');
 assert.equal(all.filter(x=>x.status==='rejected').length,1);
 saved=await store.read(home);assert.equal(saved.revision,2);
 const previous=await store.readPrevious(home);
 assert.equal(previous.revision,1,'Previous acknowledged snapshot must remain recoverable');
 assert.deepEqual(previous.snapshot,a,'Previously verified settings must survive a subsequent commit');
 const afterRestart=createSettingsRevisionStore(fs);
 assert.equal((await afterRestart.read(home)).revision,2);
 const file=path.join(home,'vecthare-settings-revisions','settings-v1.json');
 const tampered=JSON.parse(await fs.readFile(file,'utf8'));tampered.snapshot.top_k=923;
 await fs.writeFile(file,JSON.stringify(tampered));
 await assert.rejects(afterRestart.read(home),/checksum mismatch/i);
 console.log('PASS: durable settings revisions, stale writes, concurrent CAS, restart and tamper detection');
}finally{await fs.rm(home,{recursive:true,force:true});}
