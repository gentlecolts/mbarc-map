use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};

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
		//TODO: improve this loop further.  maybe iters like find?
		for (i, item) in self.data.iter_mut().enumerate() {
			if item.is_none() {
				let _ = item.insert(val);
				stored_offset = i;
				break;
			}
			//TODO: if we get here and i is last element, we should maybe panic with std::unreachable
		}
		self.free_space -= 1;

		stored_offset
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

	fn as_absolute_index(&self) -> usize {
		self.absolute_index
	}
	pub(crate) fn from_absolute_index(absolute_index: usize) -> Self {
		FaVecIndex { absolute_index }
	}
}

pub(crate) struct FaVec<T, const BLOCK_SIZE: usize> {
	//TODO: we use Box to keep data_blocks small and allow efficient resize, however we also only add to/remove from data_blocks at the end index.
	//TODO: Determine if it's most efficient to a) keep as-is, b) remove the Box, or c) remove empty blocks from the middle as well
	data_blocks: Vec<Box<DataBlock<T, BLOCK_SIZE>>>,

	//maps free space to a set of indexes (in data)
	free_space_map: BTreeMap<usize, BTreeSet<usize>>,
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
			data_blocks: vec![Box::new(DataBlock::new())],
			free_space_map: initial_map,
		}
	}

	fn update_block_free_spaces(&mut self, block_index: usize, new_free_space: usize) {
		assert!(block_index < self.data_blocks.len());
		assert!(new_free_space <= BLOCK_SIZE);

		//println!("updating block {} free space to {}", block_index, new_free_space);

		assert!(block_index < self.data_blocks.len());
		let block = self.data_blocks[block_index].as_mut();
		let old_free_space = block.free_space;

		//println!("block {} previously had {} free indices", block_index, old_free_space);

		assert!(self.free_space_map.contains_key(&old_free_space));
		let old_free_blocks = self.free_space_map.get_mut(&old_free_space).unwrap();
		assert!(!old_free_blocks.is_empty());
		assert!(old_free_blocks.contains(&block_index));
		old_free_blocks.remove(&block_index);

		if old_free_blocks.is_empty() {
			self.free_space_map.remove(&old_free_space);

			//println!("block {} was the last block with {} free indices, removing", block_index, old_free_space);
		}

		match self.free_space_map.entry(new_free_space) {
			Entry::Occupied(mut entry) => {
				entry.get_mut().insert(block_index);

				//println!("block {} was added to set with {} free indices", block_index, new_free_space);
			}
			Entry::Vacant(entry) => {
				let mut new_set = BTreeSet::new();
				new_set.insert(block_index);
				entry.insert(new_set);

				//println!("block {} was added to new set with {} free indices", block_index, new_free_space);
			}
		}

		block.free_space = new_free_space;
	}

	fn get_or_create_block_for_insert(&mut self) -> usize {
		//the current strategy here is to first, find the block(s) with the least amount of non-zero free space remaining
		//we want to concentrate data to "hot" blocks, where there's already lots of other data, so we have fewer gaps
		//from these blocks, we then select the lowest-index block, so that high index blocks are more likely to empty, and thus be dropped later during remove()

		let lowest_free_space = self.free_space_map.keys().find(|key| **key > 0);

		//if no blocks with more than 0 free space, need to make a new block
		let lowest_free_space = match lowest_free_space {
			Some(lowest_free_space) => {
				//println!("lowest has {} free blocks", lowest_free_space);
				*lowest_free_space
			}
			None => {
				if self.data_blocks.len() == Self::MAX_BLOCK_COUNT {
					//TODO: technically we can avoid this
					//consider instead using a struct that wraps the block index + block number, rather than packing them together at the end
					//this, however, comes at a cost of having a more complex type for hashing purposes
					std::process::abort(); //TODO: is abort the right exit here?
				}

				//add new data block
				let new_index = self.data_blocks.len();
				self.data_blocks.push(Box::new(DataBlock::new()));

				//track the new block's free space in our map
				let mut new_set = BTreeSet::new();
				new_set.insert(new_index);
				self.free_space_map.insert(BLOCK_SIZE, new_set);

				//we now have a block that's completely empty
				BLOCK_SIZE
			}
		};

		//pick the lowest-indexed block (with the least free space)
		//println!("{:?}", self.free_space_map);
		let possible_blocks = self.free_space_map.get_mut(&lowest_free_space).unwrap();
		assert!(!possible_blocks.is_empty());
		let block_index = possible_blocks.pop_first().unwrap();

		//if no more blocks have lowest_free_space left, then remove the index
		if possible_blocks.is_empty() {
			self.free_space_map.remove(&lowest_free_space);
		}

		//println!("{:?}", self.free_space_map);
		assert!(block_index < self.data_blocks.len());
		block_index
	}

	//TODO: insert?
	pub fn push(&mut self, val: T) -> FaVecIndex<BLOCK_SIZE> {
		let block_index = self.get_or_create_block_for_insert();

		//find an index in our block to insert the new data
		let data_block = self.data_blocks.get_mut(block_index).unwrap();
		let stored_offset = data_block.insert(val);

		//store the block's index back in its proper place in the free space map
		match self.free_space_map.get_mut(&data_block.free_space) {
			Some(block_indexes) => {
				block_indexes.insert(block_index);
			}
			None => {
				let mut new_indexes = BTreeSet::new();
				new_indexes.insert(block_index);

				self.free_space_map
					.insert(data_block.free_space, new_indexes);
			}
		}

		//println!("added item, free space map is now: {:?}", self.free_space_map);

		FaVecIndex::index_from_block_offset(block_index, stored_offset)
	}

	pub fn get(&self, index: &FaVecIndex<BLOCK_SIZE>) -> Option<&T> {
		let (block_index, offset) = FaVecIndex::index_to_block_offset(index);
		//let (block_index, offset) = (index.block_index,index.offset);

		if block_index >= self.data_blocks.len() || offset >= BLOCK_SIZE {
			return None;
		}

		match self.data_blocks.get(block_index) {
			Some(block) => block.data[offset].as_ref(),
			None => None,
		}
	}

	//todo: evaluate necessity of mut
	pub fn get_mut(&mut self, index: &FaVecIndex<BLOCK_SIZE>) -> Option<&mut T> {
		let (block_index, offset) = FaVecIndex::index_to_block_offset(index);

		if block_index >= self.data_blocks.len() || offset >= BLOCK_SIZE {
			return None;
		}

		match self.data_blocks.get_mut(block_index) {
			Some(block) => block.data[offset].as_mut(),
			None => None,
		}
	}

	fn prune_end_blocks(&mut self) {
		if let Some(empty_blocks) = self.free_space_map.get_mut(&BLOCK_SIZE) {
			//keep at least one block allocation, but otherwise we can remove empty blocks from the end
			while self.data_blocks.len() > 1
				&& self.data_blocks.last().unwrap().free_space == BLOCK_SIZE
			{
				self.data_blocks.pop();

				//new length is the index of the popped element
				//equivalent to using len()-1 before the above pop as long as len>0, len cannot be zero due to loop conditions
				let old_end_index = self.data_blocks.len();

				assert!(!empty_blocks.is_empty());
				empty_blocks.remove(&old_end_index);
			}

			if empty_blocks.is_empty() {
				self.free_space_map.remove(&BLOCK_SIZE);
			}
		};
	}

	//todo: evaluate necessity of mut
	pub fn remove(&mut self, index: &FaVecIndex<BLOCK_SIZE>) -> Option<T> {
		let (block_index, offset) = FaVecIndex::index_to_block_offset(index);

		assert!(block_index < self.data_blocks.len());
		assert!(offset < BLOCK_SIZE);

		if block_index >= self.data_blocks.len() || offset >= BLOCK_SIZE {
			//TODO: consider panic instead
			return None;
		}

		let removed_item = match self.data_blocks.get_mut(block_index) {
			Some(block) => {
				let removed = block.data.get_mut(offset).unwrap().take();

				if removed.is_some() {
					let _old_free_space = block.free_space; //for debugging

					let new_free_space = block.free_space + 1;
					self.update_block_free_spaces(block_index, new_free_space);

					assert!(self.free_space_map.contains_key(&new_free_space));
					assert!(!self.free_space_map[&new_free_space].is_empty());

					//TODO: while this should keep the map more compact, it also may cause more allocation/deallocation, which can have a cost
					//testing seems to indicate within-margin runtime, but keep a close eye on this one
					self.prune_end_blocks();
				}

				removed
			}
			None => None,
		};

		//println!("removed item, free space map is now: {:?}", self.free_space_map);
		removed_item
	}

	pub fn capacity(&self) -> usize {
		self.data_blocks.len() * BLOCK_SIZE
	}

	pub fn len(&self) -> usize {
		self.free_space_map
			.iter()
			.fold(0, |acc, (free_space, items)| {
				let used_space = BLOCK_SIZE - free_space;
				let block_count = items.len();

				acc + used_space * block_count
			})
	}

	pub fn iter(&self) -> FaVecIter<T, BLOCK_SIZE> {
		FaVecIter {
			block_index: 0,
			offset: 0,
			real_count: 0,
			fa_vec: self,
		}
	}
}

pub struct FaVecIter<'a, T, const BLOCK_SIZE: usize> {
	block_index: usize,
	offset: usize,
	real_count: usize,

	fa_vec: &'a FaVec<T, BLOCK_SIZE>,
}

impl<'a, T, const BLOCK_SIZE: usize> FaVecIter<'a, T, BLOCK_SIZE> {
	fn increment(&mut self) {
		self.offset += 1;
		if self.offset >= BLOCK_SIZE {
			self.offset = 0;
			self.block_index += 1;
		}
	}
}

impl<'a, T, const BLOCK_SIZE: usize> Iterator for FaVecIter<'a, T, BLOCK_SIZE> {
	type Item = &'a T;

	fn next(&mut self) -> Option<Self::Item> {
		while self.block_index < self.fa_vec.data_blocks.len() {
			let block_ref = self.fa_vec.data_blocks.get(self.block_index).unwrap();
			let item = &block_ref.data[self.offset];

			self.increment();

			if item.is_some() {
				self.real_count += 1;
				return item.as_ref();
			}
		}
		None
	}

	fn size_hint(&self) -> (usize, Option<usize>) {
		//TODO: cache len?
		let size = self.fa_vec.len() - self.real_count;
		(size, Some(size))
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
		let mut vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();
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
		validate_vec_properties(&mut vec);
	}

	#[test]
	fn fa_vec_properties_test_add_remove_add() {
		const ITEM_COUNT: usize = 100000;

		let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
		let mut vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();
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
		validate_vec_properties(&mut vec);
	}

	#[test]
	fn fa_vec_properties_test_add_remove_repeat() {
		//randomly add and remove elements in a 2:1 ratio until data reaches a certain count

		const ITEM_COUNT: usize = 100000;

		let mut rng = ChaCha8Rng::seed_from_u64(0xDEADBEEF);
		let mut vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();
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
		validate_vec_properties(&mut vec);
	}

	#[test]
	fn iterator_basic() {
		let mut vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();

		let _first = vec.push(5);
		let _second = vec.push(7);
		let _third = vec.push(2);

		vec.remove(&_second);

		let vals = vec.iter().copied().collect::<Vec<i64>>();

		assert_eq!(vals, vec![5, 2]);
		assert_eq!(vec.len(), 2);
		validate_vec_properties(&mut vec);
	}

	#[test]
	fn ensure_correct_prune_on_block_edge() {
		let mut vec = FaVec::<i64, TEST_BLOCK_SIZE>::new();

		let mut last_block = FaVecIndex::from_absolute_index(0);
		for i in 0..(TEST_BLOCK_SIZE + 1) {
			last_block = vec.push(i as i64);
		}

		//previously, pruning was not correctly occurring when the end block became empty and no other blocks were vacant
		vec.remove(&last_block);
		//the free space map index tracking "completely vacant" blocks was being left as an empty set, instead of removed entirely, by the prune pass
		//the application would then panic when trying to insert, due to not expecting a state where the set existed but was empty when later trying to insert
		//the logic for finding where to insert expects that all tracked vacancy counts have at least one associated block, and becomes more complicated if it has to handle empty sets
		//TODO: consider explicit test for auditing this assumption in the validate function below, though not sure it's relevant to other cases
		vec.push((TEST_BLOCK_SIZE + 1) as i64);

		assert_eq!(vec.len(), TEST_BLOCK_SIZE + 1);
		validate_vec_properties(&mut vec);
	}

	fn validate_vec_properties(vec: &mut FaVec<i64, TEST_BLOCK_SIZE>) {
		//assert that actual data matches free_space
		//assert that free space map matches free_space
		//statistically, most blocks should be pretty full, or pretty empty
		//statistically, full blocks should be mostly low-index blocks

		let mut free_space_buckets = BTreeMap::new();
		for (block_index, block) in vec.data_blocks.iter().enumerate() {
			let total_free_blocks = block
				.data
				.iter()
				.filter(|item| -> bool { item.is_none() })
				.count();
			assert_eq!(total_free_blocks, block.free_space);

			assert!(vec.free_space_map.contains_key(&block.free_space));
			assert!(vec.free_space_map[&block.free_space].contains(&block_index));

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

		println!("free space map: {:?}", vec.free_space_map);
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
