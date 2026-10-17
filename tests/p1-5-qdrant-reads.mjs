/** Qdrant list failures must be errors, not false empty collections. */
import assert from 'node:assert/strict';
import qdrant from '../qdrant-backend.js';
qdrant.baseUrl='http://mock-qdrant';
let passes=0;
const test=async(name,fn)=>{await fn();passes++;console.log('PASS',name)};
await test('unscoped Qdrant scroll HTTP 500 throws',async()=>{
 qdrant._request=async(method,path)=>{
  if(method==='GET'&&path==='/collections')return {result:{collections:[{name:'vectfox_main'}]}};
  throw Error('500 Internal Server Error');
 };
 await assert.rejects(qdrant.listItems('vectfox_main'),/500 Internal Server Error/);
});
await test('malformed Qdrant collection listing does not look empty',async()=>{
 qdrant._request=async()=>({result:{}});
 await assert.rejects(qdrant.listItems('vectfox_main'),/Malformed Qdrant collections/);
});
await test('malformed Qdrant scroll page does not look empty',async()=>{
 qdrant._request=async(method,path)=>
  method==='GET'?{result:{collections:[{name:'vectfox_main'}]}}:{result:{wrongField:[]}};
 await assert.rejects(qdrant.listItems('vectfox_main'),/Malformed Qdrant scroll/);
});
await test('verified absent collection is genuinely empty',async()=>{
 qdrant._request=async()=>({result:{collections:[]}});
 assert.deepEqual(await qdrant.listItems('vectfox_main'),[]);
});
await test('shared Qdrant dimensional incompatibility cannot erase collection',async()=>{
 const calls=[];
 qdrant._request=async(method,path,body)=>{
  calls.push({method,path,body});
  if(method==='GET'&&path==='/collections')return {result:{collections:[{name:'vectfox_main'}]}};
  if(method==='GET')return {result:{config:{params:{vectors:{size:768}}}}};
  throw Error('unexpected modification');
 };
 await assert.rejects(qdrant.insertVectors('vectfox_main',[{hash:1,text:'hello',vector:new Array(1024).fill(.2)}],{content_type:'chat'}),/Dimension mismatch/);
 assert.ok(calls.every(c=>c.method==='GET'));
});
console.log(`\n${passes} Qdrant regressions passed`);
