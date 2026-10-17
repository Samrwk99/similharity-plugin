import assert from 'node:assert/strict';
import { createCollectionMutationQueue, pluginMutationKey } from '../collection-mutation-queue.js';
const queue=createCollectionMutationQueue();
const events=[];
const release1=await queue.acquire('same');
const next=queue.acquire('same').then(release=>{events.push('second');release()});
await Promise.resolve();
assert.deepEqual(events,[],'same-collection HTTP mutations must not overlap');
const releaseUnrelated=await queue.acquire('different');
releaseUnrelated();
release1();
await next;
assert.deepEqual(events,['second']);
assert.equal(queue.pendingKeys,0);
const req=(path, backend, collectionId)=>({path,body:{backend,collectionId},user:{directories:{vectors:'/data/user/vectors'}}});
assert.equal(pluginMutationKey(req('/chunks/insert','qdrant','physicalA')),
  pluginMutationKey(req('/chunks/purge','qdrant','physicalA')));
assert.notEqual(pluginMutationKey(req('/chunks/insert','qdrant','physicalA')),
  pluginMutationKey(req('/chunks/insert','qdrant','physicalB')));
assert.equal(pluginMutationKey(req('/chunks/list','qdrant','physicalA')),null);
assert.notEqual(pluginMutationKey({path:'/collection/profile/set',body:{key:'qdrant:src:profileA'},user:{directories:{vectors:'/data/user/vectors'}}}),null);
assert.equal(pluginMutationKey({path:'/collection/profile/set',body:{key:'qdrant:src:profileA'},user:{directories:{vectors:'/data/user/vectors'}}}), pluginMutationKey({path:'/collection/profile/manifest/patch',body:{key:'qdrant:src:profileA'},user:{directories:{vectors:'/data/user/vectors'}}}));
console.log('PASS single process collection mutation serialization, independent readers and collection scoping');
