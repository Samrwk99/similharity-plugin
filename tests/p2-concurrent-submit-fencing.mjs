/** P2 #60: competing server processes cannot both start/queue same resource. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createSharedOperationStore } from '../shared-operation-store.js';
import { pluginMutationKey } from '../collection-mutation-queue.js';
const root=await fs.mkdtemp(path.join(os.tmpdir(),'vh-submit-'));
try {
 const storeA=createSharedOperationStore(fs),storeB=createSharedOperationStore(fs);
 const user=path.join(root,'user');await fs.mkdir(user);
 const races=await Promise.allSettled(Array.from({length:12},(_,i)=>
     (i%2?storeA:storeB).start(user,{resourceKey:'same-chat',category:'chat-write',label:'Duplicate'})));
 const successes=races.filter(x=>x.status==='fulfilled');
 assert.equal(successes.length,1,'only one same-target submission can claim a slot');
 assert.equal((await storeB.list(user)).operations.length,1);
 const implicitUser=path.join(root,'implicit');await fs.mkdir(implicitUser);
 const write=await storeA.beginMutation(implicitUser,null,null);
 assert.equal((await storeB.list(implicitUser)).operations.length,1);
 await assert.rejects(storeB.beginMutation(implicitUser,null,null),/another backend job|running|job queue|different VectHare job/i);
 await write();
 assert.equal((await storeB.list(implicitUser)).operations.length,0);
 const secondUser=path.join(root,'user2');await fs.mkdir(secondUser);
 const job=(await storeA.start(secondUser,{resourceKey:'A'}));
 const queued=(await storeB.start(secondUser,{resourceKey:'B'}));
 const outcomes=await Promise.allSettled([
    storeA.cancel(secondUser,queued.id),
    storeB.complete(secondUser,job.id,job.token)
 ]);
 assert.ok(outcomes.some(x=>x.status==='fulfilled'));
 const list=await storeA.list(secondUser);
 assert.ok(list.operations.length<=1,'cancellation/completion race must not duplicate an active job');
 for(const p of ['/chunks/ensure-eventbase-indexes','/backend/init/qdrant','/chunks/insert']){
    const key=pluginMutationKey({path:p,body:{collectionId:'abc'},params:{backend:'qdrant'},user:{directories:{root:'/tmp/user'}}});
    assert.ok(key,`${p} must be operation-fenced`);
 }
 console.log('PASS concurrent same-target submissions, cancellation/complete serialization and administrative mutation-route fencing');
}finally{await fs.rm(root,{recursive:true,force:true});}
