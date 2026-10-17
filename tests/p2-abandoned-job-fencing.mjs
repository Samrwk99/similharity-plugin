/** P2 #60: deterministic orphan recovery, native-write markers and old-token fencing. */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import {createSharedOperationStore} from '../shared-operation-store.js';
let now=10_000_000;
const A=createSharedOperationStore(fs,{clock:()=>now});
const B=createSharedOperationStore(fs,{clock:()=>now});
const root=await fs.mkdtemp(path.join(os.tmpdir(),'vecthare-safe-recovery-'));
try{
    const running=await A.start(root,{label:'Original native writer',category:'native-write',resourceKey:'chat:a'});
    const waiting=await B.start(root,{label:'Next chat',category:'chat-write',resourceKey:'chat:b'});
    // A native ST request must leave an in-flight marker that survives lost clients.
    const {markerId}=await A.beginNative(root,running.id,running.token);
    now+=130_000;
    await assert.rejects(B.recover(root,running.id,'invalid phrase'),/two-stage/);
    const prepared=await B.recover(root,running.id,'FENCE INTERRUPTED JOB');
    await assert.rejects(B.recover(root,running.id,'RECONCILED AND RELEASE'),/in flight/);
    assert.equal((await B.status(root,waiting.id,waiting.token)).phase,'queued');
    await A.endNative(root,running.id,running.token,markerId);
    assert.equal(prepared.fenced,true);
    assert.equal((await B.list(root)).operations[0].recoveryPrepared,true);
    await assert.rejects(A.heartbeat(root,running.id,running.token),/finalisation is blocked/);
    await assert.rejects(A.beginNative(root,running.id,running.token),/fenced/);
    await assert.rejects(A.verifyMutation(root,running.id,running.token),/fenced/);
    await assert.rejects(B.start(root,{label:'Bypass attempt',queue:false}),/Another backend job/);
    assert.equal((await B.status(root,waiting.id,waiting.token)).phase,'queued');
    const result=await B.recover(root,running.id,'RECONCILED AND RELEASE');
    assert.equal(result.recovered,true);
    assert.equal(result.requiresReconciliation,true);
    assert.equal((await B.status(root,waiting.id,waiting.token)).phase,'running');
    assert.equal((await B.status(root,running.id,running.token)).phase,'interrupted');
    await assert.rejects(A.beginNative(root,running.id,running.token),/different VectHare job/);
    await B.complete(root,waiting.id,waiting.token);
    const recent=await A.start(root,{label:'Recently online writer'});
    await assert.rejects(B.recover(root,recent.id,'FENCE INTERRUPTED JOB'),/recently active/);
    await A.complete(root,recent.id,recent.token);
    console.log('PASS interrupted native writes block recovery; explicit two-phase fencing revokes old owner and preserves queue until reconciliation');
}finally{await fs.rm(root,{recursive:true,force:true});}
