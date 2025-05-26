use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Mutex;

//TODO: most asserts in this should probably be converted to debug_assert

struct DataBlock<T, const BLOCK_SIZE: usize> {
	free_space: usize,
	data: [Option<T>; BLOCK_SIZE],
}

impl<T, const BLOCK_SIZE: usize> DataBlock<T, BLOCK_SIZE> {
	const EMPTY_ELEMENT: Option<T> = None;

	fn new() -> Self {
		Self {
			free_space: BLOCK_SIZE,
			data: [Self::EMPTY_ELEMENT; BLOCK_SIZE],
		}
	}

	fn insert(&mut self, val: T) -> usize {
		assert!(self.free_space > 0);

		//the above assert means this MUST find a value, assuming free_space is accurate
		let mut stored_offset = 0;
		let last_index = self.data.len() - 1;
		//TODO: improve this loop further.  maybe iters like find?
		for (i, item) in self.data.iter_mut().enumerate() {
			if item.is_none() {
				let _ = item.insert(val);
				stored_offset = i;
				break;
			}
			if i == last_index {
				std::unreachable!();
			}
		}
		self.free_space -= 1;

		stored_offset
	}

	fn remove(&mut self, index: usize) -> Option<T> {
		let removed = self.data.get_mut(index).unwrap().take();

		if removed.is_some() {
			self.free_space += 1;
		}

		removed
	}
}

//#[derive(Copy, Clone)]
pub(crate) struct FaVecIndex<const BLOCK_SIZE: usize> {
	absolute_index: usize,
	//TODO: can store pieces individually, determine if this is worth doing
	//block_index:usize,
	//offset:usize,
}

impl<const BLOCK_SIZE: usize> FaVecIndex<BLOCK_SIZE> {
	fn index_from_block_offset(block_index: usize, offset: usize) -> Self {
		FaVecIndex {
			absolute_index: block_index * BLOCK_SIZE + offset,
		}
	}

	fn index_to_block_offset(&self) -> (usize, usize) {
		(
			self.absolute_index / BLOCK_SIZE,
			self.absolute_index % BLOCK_SIZE,
		)
	}

	pub(crate) fn as_absolute_index(&self) -> usize {
		self.absolute_index
	}
	pub(crate) fn from_absolute_index(absolute_index: usize) -> Self {
		FaVecIndex { absolute_index }
	}
}

#[derive(Default)]
struct FreeSpaceMap<const BLOCK_SIZE: usize> {
	//maps remaining free blocks to the (set of) indices of corresponding data blocks
	map: BTreeMap<usize, BTreeSet<usize>>,
	tracked_block_count: usize,
}

impl<const BLOCK_SIZE: usize> FreeSpaceMap<BLOCK_SIZE> {
	/// Identifies the best-available data block for inserting a new element, updates the size of that element in this map
	/// if there are no free blocks, this will store current_data_block_count in the map and return it, indicating the need for a new block
	/// returns (index of data block, current free space of block)
	fn get_most_suitable_block_index_for_push(&mut self) -> (usize, usize) {
		//the current strategy here is to first, find the block(s) with the least amount of non-zero free space remaining
		//we want to concentrate data to "hot" blocks, where there's already lots of other data, so we have fewer gaps
		//from these blocks, we then select the lowest-index block, so that high index blocks are more likely to empty, and thus be dropped later during remove()

		let lowest_free_space = self.map.keys().find(|key| **key > 0);

		//if no blocks with more than 0 free space, need to make a new block
		let lowest_free_space = match lowest_free_space {
			Some(lowest_free_space) => {
				//println!("lowest has {} free blocks", lowest_free_space);
				*lowest_free_space
			}
			None => {
				//add new data block
				let new_index = self.tracked_block_count;

				//track the new block's free space in our map
				let new_set = BTreeSet::from([new_index]);
				self.map.insert(BLOCK_SIZE, new_set);
				self.tracked_block_count += 1;

				//we now have a block that's completely empty
				BLOCK_SIZE
			}
		};

		//pick the lowest-indexed block (with the least free space)
		//println!("{:?}", self.free_space_map);
		let possible_blocks = self.map.get_mut(&lowest_free_space).unwrap();
		assert!(!possible_blocks.is_empty());
		let block_index = possible_blocks.first().unwrap();

		//println!("{:?}", self.free_space_map);
		assert!(block_index < &self.tracked_block_count);
		(*block_index, lowest_free_space)
	}

	fn update_block_free_space(
		&mut self,
		block_index: usize,
		old_free_space: usize,
		new_free_space: usize,
	) {
		let possible_blocks = self.map.get_mut(&old_free_space).unwrap();
		assert!(!possible_blocks.is_empty());
		possible_blocks.remove(&block_index);

		//if no more blocks have lowest_free_space left, then remove the index
		if possible_blocks.is_empty() {
			self.map.remove(&old_free_space);
		}

		match self.map.entry(new_free_space) {
			Entry::Vacant(entry) => {
				entry.insert(BTreeSet::from_iter([block_index]));
			}
			Entry::Occupied(entry) => {
				entry.into_mut().insert(block_index);
			}
		}
	}

	/// Remove the highest index item from this map
	/// this should correspond to the last data block, which should only be popped when empty
	/// therefore, we can assert that we only pop from the set of items with BLOCK_SIZE free space
	//TODO: to safely become lock-free, we may simply never remove allocated blocks, at which point this can be removed
	fn pop(&mut self) -> usize {
		let end_block_index = self.tracked_block_count - 1;

		assert!(self.map.contains_key(&BLOCK_SIZE));
		let free_indices = self.map.get_mut(&BLOCK_SIZE).unwrap();

		assert!(free_indices.contains(&end_block_index));
		free_indices.remove(&end_block_index);

		end_block_index
	}
}

//TODO: once mutex working, remove mutex on data_blocks and introduce thread-safe resizing array.  this array should have push/pop/get, no remove, only needs to grow capacity and never has to shrink
//TODO: see AtomicPtr
pub(crate) struct FaVec<T, const BLOCK_SIZE: usize> {
	//TODO: we use Box to keep data_blocks small and allow efficient resize, however we also only add to/remove from data_blocks at the end index.
	//TODO: Determine if it's most efficient to a) keep as-is, b) remove the Box, or c) remove empty blocks from the middle as well, rather than just the end
	//TODO: keep in mind that, if Box is removed here, we still must ensure that individual elements are pointer safe, ie never moved in memory
	data_blocks: Mutex<Vec<Box<DataBlock<T, BLOCK_SIZE>>>>,

	//maps free space to a set of indexes (in data)
	free_space_map: Mutex<FreeSpaceMap<BLOCK_SIZE>>,
}

impl<T, const BLOCK_SIZE: usize> FaVec<T, BLOCK_SIZE> {
	//need to ensure block*blocksize+offset<=usize::MAX -1
	//max for offset is blocksize-1
	//block*blocksize+blocksize-1 <= usize::max - 1
	//block*(blocksize+1) <= usize::max
	//block <= usize::max / (blocksize+1)
	//if block = usize::max/(blocksize+1) and we are full, push should panic instead of adding new data
	const MAX_BLOCK_COUNT: usize = usize::MAX / (BLOCK_SIZE + 1);

	pub fn new() -> Self {
		let mut initial_set = BTreeSet::new();
		initial_set.insert(0usize);
		let mut initial_map = BTreeMap::new();
		initial_map.insert(BLOCK_SIZE, initial_set);

		Self {
			data_blocks: Mutex::new(vec![Box::new(DataBlock::new())]),
			free_space_map: Default::default(),
		}
	}

	//TODO: rename to insert?
	pub fn push(&self, val: T) -> FaVecIndex<BLOCK_SIZE> {
		let mut data_blocks_lock = self.data_blocks.lock().unwrap();
		let mut free_space_lock = self.free_space_map.lock().unwrap();

		let (push_block_index, free_space_in_block) =
			free_space_lock.get_most_suitable_block_index_for_push();
		free_space_lock.update_block_free_space(
			push_block_index,
			free_space_in_block,
			free_space_in_block - 1,
		);

		let current_data_block_count = data_blocks_lock.len();
		assert!(push_block_index <= current_data_block_count);

		if push_block_index == current_data_block_count {
			data_blocks_lock.push(Box::new(DataBlock::new()));
		}

		let new_block = data_blocks_lock.get_mut(push_block_index).unwrap();
		assert_eq!(free_space_in_block, new_block.free_space);

		let index = new_block.insert(val);
		FaVecIndex::index_from_block_offset(push_block_index, index)
	}

	pub(crate) unsafe fn get_raw(&self, index: &FaVecIndex<BLOCK_SIZE>) -> Option<*mut T> {
		let (block_index, offset) = FaVecIndex::index_to_block_offset(index);

		let mut data_blocks_lock = self.data_blocks.lock().unwrap();

		if block_index >= data_blocks_lock.len() || offset >= BLOCK_SIZE {
			return None;
		}

		match data_blocks_lock.get_mut(block_index) {
			Some(block) => block.data[offset].as_mut().map(|val| val as *mut T),
			None => None,
		}
	}

	pub fn remove(&self, index: &FaVecIndex<BLOCK_SIZE>) -> Option<T> {
		let mut data_blocks_lock = self.data_blocks.lock().unwrap();
		let mut free_space_lock = self.free_space_map.lock().unwrap();

		let (block_index, offset) = index.index_to_block_offset();

		assert!(block_index < data_blocks_lock.len());
		assert!(offset < BLOCK_SIZE);

		let removed_item = match data_blocks_lock.get_mut(block_index) {
			Some(block) => {
				let old_free_space = block.free_space;
				let removed = block.remove(offset);

				if removed.is_some() {
					free_space_lock.update_block_free_space(
						block_index,
						old_free_space,
						old_free_space + 1,
					);
				}

				removed
			}
			None => None,
		};

		//println!("removed item, free space map is now: {:?}", self.free_space_map);
		removed_item
	}

	pub fn capacity(&self) -> usize {
		self.data_blocks.lock().unwrap().len() * BLOCK_SIZE
	}

	pub fn len(&self) -> usize {
		self.free_space_map
			.lock()
			.unwrap()
			.map
			.iter()
			.fold(0, |acc, (free_space, items)| {
				let used_space = BLOCK_SIZE - free_space;
				let block_count = items.len();

				acc + used_space * block_count
			})
	}
}

#[cfg(test)]
mod tests {
	use std::collections::BTreeMap;

	use rand::prelude::*;
	use rand_chacha::ChaCha8Rng;

	use crate::fixed_address_continuous_allocation::{FaVec, FaVecIndex};

	const TEST_BLOCK_SIZE: usize = 512;

	#[test]
	fn fa_vec_properties_test_random_io() {
		//randomly add and remove elements in a 2:1 ratio until data reaches a certain count

		const ITEM_COUNT: usize = 100000;

		let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
		let vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();
		let mut keys = Vec::<FaVecIndex<TEST_BLOCK_SIZE>>::new();

		//note, this access pattern is pretty basic, as it is very unlikely to leave holes in the vec
		while keys.len() < ITEM_COUNT {
			let remove_element = rng.random_ratio(1, 3);

			if remove_element && !keys.is_empty() {
				let key_index = rng.random_range(0..keys.len());
				let key = keys.swap_remove(key_index);
				vec.remove(&key);
			} else {
				let new_value = rng.random::<i64>();
				let new_key = vec.push(new_value);
				keys.push(new_key);
			}
		}

		//this output should visually prove there's a random-ish (But generally increasing) order to inserted indexes
		//it's noisy, but worth looking at if there's an issue
		//println!("indices inserted: {:?}",keys);

		assert_eq!(vec.len(), keys.len());
		validate_vec_properties(&vec);
	}

	#[test]
	fn fa_vec_properties_test_add_remove_add() {
		const ITEM_COUNT: usize = 100000;

		let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
		let vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();
		let mut keys = Vec::<FaVecIndex<TEST_BLOCK_SIZE>>::new();

		const INITIAL_ITEM_COUNT: usize = ITEM_COUNT * 2;
		for total_items in 0..INITIAL_ITEM_COUNT {
			let new_value = rng.random::<i64>();
			let new_key = vec.push(new_value);
			keys.push(new_key);

			assert_eq!(
				vec.capacity(),
				TEST_BLOCK_SIZE * (1 + total_items / TEST_BLOCK_SIZE)
			);
		}

		const HALF_ITEM_COUNT: usize = ITEM_COUNT / 2;
		for _ in 0..(ITEM_COUNT + HALF_ITEM_COUNT) {
			let key_index = rng.random_range(0..keys.len());
			let key = keys.swap_remove(key_index);
			vec.remove(&key);

			//TODO: it's impossible to assess capacity in-loop, due to the random remove order.  Adjust/implement test(s) to account for this
		}

		for _ in 0..HALF_ITEM_COUNT {
			let new_value = rng.random::<i64>();
			let new_key = vec.push(new_value);
			keys.push(new_key);
		}

		//this output should visually prove there's a random-ish (But generally increasing) order to inserted indexes
		//it's noisy, but worth looking at if there's an issue
		//println!("indices inserted: {:?}",keys);

		assert_eq!(vec.len(), keys.len());
		validate_vec_properties(&vec);
	}

	#[test]
	fn fa_vec_properties_test_add_remove_repeat() {
		const ITEM_COUNT: usize = 100000;

		let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
		let vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();
		let mut keys = Vec::<FaVecIndex<TEST_BLOCK_SIZE>>::new();

		for _ in 0..2 * ITEM_COUNT {
			let new_value = rng.random::<i64>();
			let new_key = vec.push(new_value);
			keys.push(new_key);
		}

		for _ in 0..10 {
			for _ in 0..ITEM_COUNT {
				let key_index = rng.random_range(0..keys.len());
				let key = keys.swap_remove(key_index);
				vec.remove(&key);
			}

			for _ in 0..ITEM_COUNT {
				let new_value = rng.random::<i64>();
				let new_key = vec.push(new_value);
				keys.push(new_key);
			}
		}

		/*
		//Using this final remove step to end with N values causes a bell curve distribution
		//This is valid as a result, but useless from a testing perspective
		//this is because I cannot choose what address to remove from, only which one I pick when pushing new data
		for i in 0..ITEM_COUNT{
			let key_index=rng.gen_range(0..keys.len());
			let key= keys.swap_remove(key_index);
			vec.remove(&key);
		}
		// */

		//this output should visually prove there's a random-ish (But generally increasing) order to inserted indexes
		//it's noisy, but worth looking at if there's an issue
		//println!("indices inserted: {:?}",keys);

		assert_eq!(vec.len(), keys.len());
		validate_vec_properties(&vec);
	}

	#[test]
	fn ensure_correct_prune_on_block_edge() {
		let vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();

		let mut last_block = FaVecIndex::from_absolute_index(0);
		for i in 0..((TEST_BLOCK_SIZE + 1) as i64) {
			last_block = vec.push(i);
		}

		//previously, pruning was not correctly occurring when the end block became empty and no other blocks were vacant
		vec.remove(&last_block);
		//the free space map index tracking "completely vacant" blocks was being left as an empty set, instead of removed entirely, by the prune pass
		//the application would then panic when trying to insert, due to not expecting a state where the set existed but was empty when later trying to insert
		//the logic for finding where to insert expects that all tracked vacancy counts have at least one associated block, and becomes more complicated if it has to handle empty sets
		//TODO: consider explicit test for auditing this assumption in the validate function below, though not sure it's relevant to other cases
		vec.push((TEST_BLOCK_SIZE + 1) as i64);

		assert_eq!(vec.len(), TEST_BLOCK_SIZE + 1);
		validate_vec_properties(&vec);
	}

	fn validate_vec_properties(vec: &FaVec<i64, TEST_BLOCK_SIZE>) {
		let data_blocks_lock = vec.data_blocks.lock().unwrap();
		let free_space_lock = vec.free_space_map.lock().unwrap();

		//assert that actual data matches free_space
		//assert that free space map matches free_space
		//statistically, most blocks should be pretty full, or pretty empty
		//statistically, full blocks should be mostly low-index blocks

		let mut free_space_buckets = BTreeMap::new();
		for (block_index, block) in data_blocks_lock.iter().enumerate() {
			let total_free_blocks = block
				.data
				.iter()
				.filter(|item| -> bool { item.is_none() })
				.count();
			assert_eq!(total_free_blocks, block.free_space);

			assert!(free_space_lock.map.contains_key(&block.free_space));
			assert!(free_space_lock.map[&block.free_space].contains(&block_index));

			match free_space_buckets.get_mut(&block.free_space) {
				Some(count) => {
					*count += 1;
				}
				None => {
					free_space_buckets.insert(block.free_space, 1usize);
				}
			}
		}

		//there should be lots of free or empty blocks, but not as many in between
		//this vec should have a bathtub curve
		//TODO: this is just a basic guess approach, but we should ideally investigate how to determine this more rigorously
		//actually, it seems like rather than forming a bathtub, we do aggressively push towards keeping blocks full
		//barring pathologic bad remove behavior, we see a decreasing trend of free space vs count
		//TODO: reassess metrics
		let free_space_counts: Vec<usize> = free_space_buckets.values().copied().collect();
		let least_free_block_count = *free_space_counts.first().unwrap();
		let most_free_block_count = *free_space_counts.last().unwrap();
		let middle_free_block_count = free_space_counts[free_space_counts.len() / 2];

		println!("free space map: {:?}", free_space_lock.map);
		println!("free count vec: {:?}", free_space_counts);
		println!(
			"least/middle/most free: {} - {} - {}",
			least_free_block_count, middle_free_block_count, most_free_block_count
		);
		assert!(least_free_block_count >= middle_free_block_count);
		//assert!(most_free_block_count >= middle_free_block_count);

		//also, we should have more full blocks than empty ones
		//TODO: linear regression over the whole list should give a negative slope, instead of this basic comparison
		assert!(least_free_block_count >= most_free_block_count);
	}
}
