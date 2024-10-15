//! Implementation of a [Minimally-blocking, Atomic Reference Counted Map](MbarcMap).
//!
//! To break that down, map at the heart of this crate achieves the following core goals:
//! - Minimally-blocking: a user should never need to wrap this map in a mutex, all internal mutexes are held for as short of a duration as possible, and there are no deadlock cases.  Users only need to manually take locks to individual elements.
//! - Atomic Reference Counted - all data stored within the map are reference counted in a thread-safe manner, and it is safe to hold these references indefinitely

pub use fixed_address_continuous_allocation::*;

pub use data_reference::*;
pub use data_reference_generic::*;
pub use minimally_blocking_atomic_reference_counted_map::*;

mod fixed_address_continuous_allocation;

mod data_holder;
mod data_reference;
mod data_reference_generic;

mod minimally_blocking_atomic_reference_counted_map;

#[cfg(test)]
mod tests {
	use std::any::TypeId;
	use std::collections::HashMap;
	use std::mem::size_of;
	use std::ops::Deref;
	use std::sync::{Arc, Mutex, MutexGuard};
	use std::thread;
	use rand::prelude::*;
	use rand_chacha::ChaCha8Rng;
	use rayon::prelude::*;

	use super::*;

	type PreSeed<const N: usize> = Box<[(i64, i64); N]>;

	const FIXED_SEED: u64 = 0xDEADBEEF;

	fn make_data_pairs<const N: usize>(seed: u64) -> PreSeed<N> {
		let mut rng = ChaCha8Rng::seed_from_u64(seed);

		let mut pairs = Box::new([(0i64, 0i64); N]);
		for i in 0..N {
			let a = rng.gen_range(i64::MIN..i64::MAX);
			let b = rng.gen_range(i64::MIN..i64::MAX);
			pairs[i] = (a, b);
		}

		pairs
	}


	#[test]
	fn test_use_element_after_drop_one_value() {
		let concurrent_hash = Arc::new(MbarcMap::new());

		let key: i64 = 2;
		let value: &str = "Hi";
		concurrent_hash.insert(key, value);

		let first_value: Option<DataReference<&str>> = concurrent_hash.get(&key);
		drop(concurrent_hash);

		let first_value: DataReference<&str> = first_value.unwrap();

		assert_eq!(first_value.ref_count(), 1);
		assert!(first_value.is_marked_deleted());

		let first_value_lock: MutexGuard<&str> = first_value.lock().unwrap();
		assert_eq!(*first_value_lock, "Hi");
	}

	#[test]
	fn test_safe_to_use_element_after_map_is_dropped() {
		const N: usize = 1000;

		let source_data = make_data_pairs::<N>(FIXED_SEED);
		let concurrent_hash = Arc::new(MbarcMap::new());

		insert_several_threaded(&source_data, &concurrent_hash);

		let first_value = concurrent_hash.get(&source_data[0].0);
		assert!(first_value.is_some());

		drop(concurrent_hash);
		let first_value = first_value.unwrap();

		//let raw_data = first_value.raw_data();
		assert_eq!(first_value.ref_count(), 1);
		assert!(first_value.is_marked_deleted());

		let first_value_lock = first_value.lock().unwrap();
		assert_eq!(*first_value_lock, source_data[0].1);
	}

	#[test]
	fn test_insert_remove_insert() {
		const STEP_SIZE: usize = 30000;
		const START_REMOVING_INDEX: usize = 2 * STEP_SIZE;
		const N: usize = 3 * STEP_SIZE;

		let source_data = make_data_pairs::<N>(FIXED_SEED);
		let mut base_hash = Box::new(HashMap::new());
		let concurrent_hash = Arc::new(MbarcMap::new());

		source_data.iter().enumerate().for_each(|(i, (k, v))| {
			if i >= START_REMOVING_INDEX {
				let removal_index = i - START_REMOVING_INDEX;
				let remove_key = source_data.get(removal_index).unwrap().0;

				base_hash.remove(&remove_key);
			}

			base_hash.insert(*k, *v);
		});

		let (initial_insertions, parallel_inserted_while_removing) = source_data.split_at(START_REMOVING_INDEX);

		for (k, v) in initial_insertions {
			concurrent_hash.insert(*k, *v);
		}

		parallel_inserted_while_removing.par_iter().enumerate().for_each(|(i, (k, v))| {
			let remove_key = source_data.get(i).unwrap().0;
			concurrent_hash.remove(&remove_key);

			concurrent_hash.insert(*k, *v);
		});

		assert_hash_contents_equal(&base_hash, concurrent_hash)
	}


	fn insert_several<const N: usize>(from: &PreSeed<N>, to: &mut HashMap<i64, i64>) {
		for (k, v) in from.iter() {
			to.insert(*k, *v);
		}
	}

	fn insert_several_threaded<const N: usize>(from: &PreSeed<N>, to: &Arc<MbarcMap<i64, i64>>) {
		from.par_iter().for_each(|(k, v)| { to.insert(*k, *v); });
	}

	#[test]
	fn test_insert_only() {
		const N: usize = 100000;

		let source_data = make_data_pairs::<N>(FIXED_SEED);
		let mut base_hash = Box::new(HashMap::new());
		let concurrent_hash = Arc::new(MbarcMap::new());

		insert_several(&source_data, &mut base_hash);
		insert_several_threaded(&source_data, &concurrent_hash);

		//println!("Confirming length after insert");
		assert_eq!(base_hash.len(), N);
		assert_eq!(concurrent_hash.len(), N);

		assert_hash_contents_equal(&base_hash, concurrent_hash);

		println!("Insert test done");
	}

	fn assert_hash_contents_equal(base_hash: &Box<HashMap<i64, i64>>, concurrent_hash: Arc<MbarcMap<i64, i64>>) {
		//println!("Comparing values after insert");
		for (k, v) in base_hash.iter() {
			//println!("Checking for {} and {}",k,v);
			assert!(concurrent_hash.contains(k));
			//println!("Key found, comparing value");

			let expected_value: i64 = *v;

			//println!("Fetching from map");
			let data_from_map = concurrent_hash.get(k).unwrap();

			//println!("Checking inner data");
			//let raw_data = data_from_map.raw_data();
			let current_ref_count = data_from_map.ref_count();
			let is_raw_deleted = data_from_map.is_marked_deleted();
			assert_eq!(current_ref_count, 2);
			assert!(!is_raw_deleted);

			//println!("making sure lock's ok");
			let data_mutex_poisoned = data_from_map.is_poisoned();
			assert!(!data_mutex_poisoned);

			//println!("Taking lock on inner data");
			//let data_lock=data_from_map.lock().unwrap();
			let data_lock = data_from_map.try_lock();
			let data_lock_ok = data_lock.is_ok();
			assert!(data_lock_ok);

			//println!("Assigning value");
			let true_lock = data_lock.unwrap();
			let actual_value = *true_lock;

			assert_eq!(expected_value, actual_value);
			//println!("Pair {}, {} passed!",k,v);

			//drop(true_lock);
		}
	}

	#[test]
	fn test_iterator() {
		const N: usize = 100000;

		let source_data = make_data_pairs::<N>(FIXED_SEED);
		let mut base_hash = Box::new(HashMap::new());
		let concurrent_hash = Arc::new(MbarcMap::new());

		insert_several(&source_data, &mut base_hash);
		insert_several_threaded(&source_data, &concurrent_hash);

		for (k, v) in concurrent_hash.iter_copied_keys() {
			assert!(base_hash.contains_key(&k));

			let base_val = base_hash.remove(&k).unwrap();
			assert_eq!(base_val, *v.lock().unwrap());
		}

		assert_eq!(base_hash.len(), 0);
	}


	#[test]
	fn test_drop() {
		const N: usize = 100000;

		let source_data = make_data_pairs::<N>(FIXED_SEED);
		let concurrent_hash = Arc::new(MbarcMap::new());

		insert_several_threaded(&source_data, &concurrent_hash);

		let iter = concurrent_hash.iter();
		drop(concurrent_hash);

		for v in iter {
			assert!(v.is_marked_deleted());
			assert_eq!(v.ref_count(), 1);
		}
	}

	trait TestTrait {
		fn get(&self) -> u64 {
			5
		}
	}

	const TEST_TYPE_VALUE: u64 = 2;

	struct TestType {}

	impl TestTrait for TestType {
		fn get(&self) -> u64 {
			TEST_TYPE_VALUE
		}
	}

	#[test]
	fn test_mutate_deref() {
		assert_eq!(size_of::<TestType>(), 0);

		let map = MbarcMap::<usize, TestType>::new();
		map.insert(0, TestType {});

		let item = map.get(&0).unwrap();
		assert_eq!(item.lock().unwrap().deref().get(), TEST_TYPE_VALUE);

		let raw: &Mutex::<dyn TestTrait> = item.deref();
		assert_eq!(raw.lock().unwrap().get(), TEST_TYPE_VALUE);
	}

	#[test]
	fn test_locked_iteration() {
		const N: usize = 1000;

		let source_data = make_data_pairs::<N>(FIXED_SEED);
		let concurrent_hash = Arc::new(MbarcMap::new());

		insert_several_threaded(&source_data, &concurrent_hash);

		let result=thread::scope(|scope|{
			let v:Arc<Mutex<Vec<(i64,DataReference<i64>)>>>=Default::default();

			for _ in 0..2{
				let my_hash=concurrent_hash.clone();
				let my_vec=v.clone();
				scope.spawn(move ||{
					for (k, v) in my_hash.iter_exclusive().iter(){
						my_vec.lock().unwrap().push((*k,v.clone()))
					}
				});
			}

			v
		});

		let result=match Arc::try_unwrap(result){
			Ok(r) => {
				r.into_inner().unwrap()
			}
			Err(_) => {
				unreachable!()
			}
		};

		assert_eq!(result.len(),2*N);

		for i in 0..N{
			let (k1,v1)=&result[i];
			let (k2,v2)=&result[i+N];

			assert_eq!(*k1,*k2);

			let v1=*v1.lock().unwrap();
			let v2=*v2.lock().unwrap();
			assert_eq!(v1,v2);
		}
	}

	struct GenericRefTestType{
		a: MbarcMap<usize,u32>,
		b: MbarcMap<usize,u64>,
	}

	impl GenericRefTestType{
		const A_ITEM_KEY: usize=0;
		const B_ITEM_KEY: usize=0;

		fn new(a_val:u32,b_val:u64)->Self{
			let a=MbarcMap::new();
			let b=MbarcMap::new();

			a.insert(Self::A_ITEM_KEY, a_val);
			b.insert(Self::B_ITEM_KEY, b_val);

			Self{a,b}
		}

		fn get_from_a(&self)->DataReferenceGeneric{
			DataReferenceGeneric::from(self.a.get(&Self::A_ITEM_KEY).unwrap())
		}

		fn get_from_b(&self)->DataReferenceGeneric{
			DataReferenceGeneric::from(self.b.get(&Self::B_ITEM_KEY).unwrap())
		}

		fn a_ref_count(&self)->usize{
			//numder of refs, minus the temporary one we just created
			self.a.get(&Self::A_ITEM_KEY).unwrap().ref_count() - 1
		}

		fn b_ref_count(&self)->usize{
			//numder of refs, minus the temporary one we just created
			self.b.get(&Self::B_ITEM_KEY).unwrap().ref_count() - 1
		}

		fn set_a(&self,value: u32){
			*self.a.get(&Self::A_ITEM_KEY).unwrap().lock().unwrap()=value;
		}

		fn set_b(&self,value: u64){
			*self.b.get(&Self::B_ITEM_KEY).unwrap().lock().unwrap()=value;
		}
	}

	#[test]
	fn test_generic_morphing(){
		const A_VALUE:u32=0;
		const B_VALUE:u64=0;

		let tester=GenericRefTestType::new(A_VALUE,B_VALUE);

		assert_eq!(tester.a_ref_count(),1);
		assert_eq!(tester.b_ref_count(),1);

		let a_generic=tester.get_from_a();
		let b_generic=tester.get_from_b();

		assert_eq!(tester.a_ref_count(),2);
		assert_eq!(tester.b_ref_count(),2);

		assert_eq!(a_generic.type_id(),TypeId::of::<DataReference<u32>>());
		assert_eq!(a_generic.inner_type_id(),TypeId::of::<u32>());

		assert_eq!(b_generic.type_id(),TypeId::of::<DataReference<u64>>());
		assert_eq!(b_generic.inner_type_id(),TypeId::of::<u64>());

		let not_a=a_generic.to_typed::<u128>();
		let not_b=b_generic.to_typed::<u128>();

		assert!(not_a.is_none());
		assert!(not_b.is_none());

		assert_eq!(tester.a_ref_count(),2);
		assert_eq!(tester.b_ref_count(),2);

		let actually_a=a_generic.to_typed::<u32>();
		let actually_b=b_generic.to_typed::<u64>();

		assert!(actually_a.is_some());
		assert!(actually_b.is_some());

		assert_eq!(tester.a_ref_count(),3);
		assert_eq!(tester.b_ref_count(),3);

		drop(a_generic);
		assert_eq!(tester.a_ref_count(),2);

		drop(b_generic);
		assert_eq!(tester.b_ref_count(),2);

		let actually_a=actually_a.unwrap();
		let actually_b=actually_b.unwrap();

		assert_eq!(*actually_a.lock().unwrap(),A_VALUE);
		assert_eq!(*actually_b.lock().unwrap(),B_VALUE);

		const NEW_A: u32=11;
		const NEW_B: u64=12;

		tester.set_a(NEW_A);
		tester.set_b(NEW_B);

		assert_eq!(*actually_a.lock().unwrap(),NEW_A);
		assert_eq!(*actually_b.lock().unwrap(),NEW_B);
	}

	#[test]
	fn test_generic_early_drop(){
		const A_VALUE:u32=0;
		const B_VALUE:u64=0;

		let tester=GenericRefTestType::new(A_VALUE,B_VALUE);

		assert_eq!(tester.a_ref_count(),1);
		assert_eq!(tester.b_ref_count(),1);

		let a_generic=tester.get_from_a();
		let b_generic=tester.get_from_b();

		assert_eq!(tester.a_ref_count(),2);
		assert_eq!(tester.b_ref_count(),2);

		drop(tester);

		let actually_a=a_generic.to_typed::<u32>();
		let actually_b=b_generic.to_typed::<u64>();

		assert!(actually_a.is_some());
		assert!(actually_b.is_some());

		let actually_a=actually_a.unwrap();
		let actually_b=actually_b.unwrap();

		assert_eq!(actually_a.ref_count(),2);
		assert_eq!(actually_b.ref_count(),2);

		drop(a_generic);
		assert_eq!(actually_a.ref_count(),1);

		drop(b_generic);
		assert_eq!(actually_b.ref_count(),1);

		assert_eq!(*actually_a.lock().unwrap(),A_VALUE);
		assert_eq!(*actually_b.lock().unwrap(),B_VALUE);
	}
}
