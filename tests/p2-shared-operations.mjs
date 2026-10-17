/** Durable cross-process operation coordinator contract tests (real filesystem). */
import assert from 'node:assert/strict';
import fs from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { createSharedOperationStore } from '../shared-operation-store.js';
const dir=await fs.mkdtemp(path.join(os.tmpdir(),'vh-op-tests-'));
const storeA=createSharedOperationStore(fs),storeB=createSharedOperationStore(fs);
let passed=0;async function test(name,fn){await fn();passed++;console.log('PASS',name);}
try{
    const root=path.join(dir,'same-user');await fs.mkdir(root);
    const A=await storeA.start(root,{label:'Chat A',resourceKey:'chat:a'});
    assert.equal(A.phase,'running');
    await test('new backend job owns the shared per-user lease',async()=>{
        assert.equal((await storeB.list(root)).operations.length,1);
        await assert.rejects(storeB.verifyMutation(root,'other','wrong'),/different VectHare job/);
        assert.equal((await storeB.verifyMutation(root,A.id,A.token)).active,true);
    });
    await test('duplicate backend jobs are coalesced instead of flooding the queue',async()=>{
        await assert.rejects(storeB.start(root,{label:'Chat A again',resourceKey:'chat:a'}),/already running or queued/);
        const jobs=await storeA.list(root);
        assert.equal(jobs.operations.length,1);
    });
    const B=await storeB.start(root,{label:'Chat B',resourceKey:'chat:b'});
    assert.equal(B.phase,'queued');
    await test('second process queues without obtaining write authority',async()=>{
        assert.equal((await storeB.status(root,B.id,B.token)).phase,'queued');
        await assert.rejects(storeB.heartbeat(root,B.id,B.token),/not active/);
        await assert.rejects(storeB.status(root,A.id,'bad'),/Invalid operation token/);
    });
    await test('remote tab requests cancellation, owner sees it at safe checkpoint',async()=>{
        await storeB.cancel(root,A.id);
        const state=await storeA.heartbeat(root,A.id,A.token,{completed:3,total:9});
        assert.equal(state.cancelRequested,true);
        await assert.rejects(storeA.verifyMutation(root,A.id,A.token),/cancellation is pending/);
    });
    await test('finishing releases only exact owner and promotes queued operation',async()=>{
        await assert.rejects(storeB.complete(root,A.id,'wrong','completed'),/token invalid/);
        await storeA.complete(root,A.id,A.token,'stopped','Cancelled safely');
        const s=await storeB.status(root,B.id,B.token);
        assert.equal(s.phase,'running');
        assert.equal((await storeB.verifyMutation(root,B.id,B.token)).active,true);
        await assert.rejects(storeA.complete(root,A.id,A.token,'completed'),/not active/);
    });
    const C=await storeA.start(root,{label:'Chat C'});
    await test('queued operation can be cancelled without touching live vectors',async()=>{
        assert.equal(C.phase,'queued');
        await storeB.cancel(root,C.id);
        assert.equal((await storeA.status(root,C.id,C.token)).phase,'stopped');
        assert.equal((await storeA.status(root,B.id,B.token)).phase,'running');
    });
    await storeB.complete(root,B.id,B.token);
    await test('orphaned active jobs cannot be stolen by timeouts',async()=>{
        const D=await storeA.start(root,{label:'Unresponsive writer'});
        assert.equal(D.phase,'running');
        const E=await storeB.start(root,{label:'Later writer'});
        assert.equal(E.phase,'queued');
        // No automatic expiry takeover even if the original browser vanished.
        assert.equal((await storeA.status(root,E.id,E.token)).phase,'queued');
        assert.equal((await storeA.list(root)).operations.length,2);
    });
    await test('operation list never exposes ownership secrets',async()=>{
        const listing=JSON.stringify(await storeB.list(root));
        assert.ok(!listing.includes(A.token));assert.ok(!listing.includes(B.token));
        assert.ok(!listing.includes('"token"'));
    });
    const inflightRoot=path.join(dir,'inflight-owner');await fs.mkdir(inflightRoot);
    await test('in-flight server writes cannot be abandoned into another writer',async()=>{
        const op=await storeA.start(inflightRoot,{label:'Slow Qdrant insert'});
        const releaseRequest=await storeA.beginMutation(inflightRoot,op.id,op.token);
        await assert.rejects(storeB.complete(inflightRoot,op.id,op.token),/still in flight/);
        const waiting=await storeB.start(inflightRoot,{label:'Next write'});
        assert.equal(waiting.phase,'queued');
        await releaseRequest();
        await storeB.complete(inflightRoot,op.id,op.token,'completed');
        assert.equal((await storeB.status(inflightRoot,waiting.id,waiting.token)).phase,'running');
    });
    const other=path.join(dir,'second-user');await fs.mkdir(other);
    await test('per-user isolation prevents other accounts seeing jobs',async()=>{
        assert.equal((await storeB.list(other)).operations.length,0);
    });
    console.log(`\n${passed} shared-operation durability and isolation checks passed`);
}finally{await fs.rm(dir,{recursive:true,force:true});}
