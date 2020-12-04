import numpy as np

class Indexer(object):
    """
    Bijection between objects and integers starting at 0. Useful for mapping
    labels, features, etc. into coordinates of a vector space.

    Attributes:
        objs_to_ints
        ints_to_objs
    """
    def __init__(self):
        self.objs_to_ints = {}
        self.ints_to_objs = {}

    def __repr__(self):
        return str([str(self.get_object(i)) for i in range(0, len(self))])

    def __str__(self):
        return self.__repr__()

    def __len__(self):
        return len(self.objs_to_ints)

    def get_object(self, index):
        """
        :param index: integer index to look up
        :return: Returns the object corresponding to the particular index or None if not found
        """
        if (index not in self.ints_to_objs):
            return None
        else:
            return self.ints_to_objs[index]

    def contains(self, object):
        """
        :param object: object to look up
        :return: Returns True if it is in the Indexer, False otherwise
        """
        return self.index_of(object) != -1

    def index_of(self, object):
        """
        :param object: object to look up
        :return: Returns -1 if the object isn't present, index otherwise
        """
        if (object not in self.objs_to_ints):
            return -1
        else:
            return self.objs_to_ints[object]

    def add_and_get_index(self, object, add=True):
        """
        Adds the object to the index if it isn't present, always returns a nonnegative index
        :param object: object to look up or add
        :param add: True by default, False if we shouldn't add the object. If False, equivalent to index_of.
        :return: The index of the object
        """
        if not add:
            return self.index_of(object)
        if (object not in self.objs_to_ints):
            new_idx = len(self.objs_to_ints)
            self.objs_to_ints[object] = new_idx
            self.ints_to_objs[new_idx] = object
        return self.objs_to_ints[object]

def readInfo(args):
    """
    Read statistics of the dataset, and create id2type.txt
    """
    f = open("/".join([args.dir, 'datainfo.md']), "r")
    print("**********\nreadInfo")
    for line in f:
        eles = line.strip().split()
        assert len(eles) == 5
        for i in range(len(eles)):
            assert eles[i].isnumeric(), "error: eles[%d] isnumeric() = False"%(i)
            eles[i] = int(eles[i])
    print("**********")
    num_movies, num_genres, num_cast, num_users, total = eles[:5]
    f.close()

    # create id2type
    f = open("/".join([args.dir, 'id2type.txt']), "w")
    id_base = 0
    for i in range(num_movies):
        f.write("%d movie\n"%(i))
    id_base += num_movies
    for i in range(num_genres):
        f.write("%d genre\n"%(i + id_base))
    id_base += num_genres
    for i in range(num_cast):
        f.write("%d cast\n"%(i + id_base))
    id_base += num_cast
    for i in range(num_users):
        f.write("%d user\n"%(i + id_base))
    id_base += num_users
    assert id_base == total

    f.close()
    return num_movies, num_genres, num_cast, num_users, total

def readEmbeddings(datadir, filename, user_n, user_base, movie_n, movie_base):

    user_emb = None
    movie_emb = None

    f = open("/".join([datadir, filename]), "r")
    for line in f:
        eles = line.strip().split()
        if len(eles) < 3:
            _, embed_dim = int(eles[0]), int(eles[1])
            user_emb = np.empty(shape=(user_n, embed_dim))
            movie_emb = np.empty(shape=(movie_n, embed_dim))
            print("readEmbeddings: create user_emb %s and movie_emb %s"%(user_emb.shape, movie_emb.shape))
            continue
        if not eles[0].isnumeric():
            continue
        node_id = int(eles[0])
        if node_id > user_base:  # user_id is the largest among four node types
            user_emb[node_id - user_base] = [float(eles[i + 1]) for i in range(embed_dim)]
        elif node_id < movie_n:
            movie_emb[node_id - movie_base] = [float(eles[i + 1]) for i in range(embed_dim)]

    print("readEmbeddings: finished")
    return user_emb, movie_emb

