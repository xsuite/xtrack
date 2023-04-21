import pathlib
import re
import numpy as np

gblmath = {"np": np}
for k, fu in np.__dict__.items():
    if type(fu) is np.ufunc:
        gblmath[k] = fu


def _to_str(arr, digits, fixed="g"):
    """covert array to string repr"""
    if arr.dtype.kind in "SU":
        return arr
    elif arr.dtype.kind in "iu":
        return np.char.mod("%d", arr)
    elif arr.dtype.kind in "f":
        fmt = "%%.%d%s" % (digits, fixed)
        return np.char.mod(fmt, arr)
    else:
        return arr.astype("U")


class Mask:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, key):
        """
        t.mask[1] -> row
        t.mask['a'] -> pattern
        l.mask[10:20]-> range
        l.mask['a':'b'] -> name range
        l.mask['a':'b':'myname'] -> name range with 'myname' column
        l.mask[-2:2:'x'] -> name value range with 'x' column
        l.mask[-2:2:'x',...] -> & combinations
        """
        mask = np.zeros(self.table._nrows, dtype=bool)
        if isinstance(key, int):
            mask = np.zeros(self.table._nrows, dtype=bool)
            mask[key] = True
            return mask
        elif isinstance(key, str):
            if self.table._offset_sep in key or self.table._count_sep in key:
                mask[:] = self[key:key] # use slice (next elif)
            else:
                mask[:] = self.table._get_name_mask(key, self.table._index)
            mask[:] = self.table._get_name_mask(key, self.table._index)
            if self.table._error_on_row_not_found and not mask.any():
                raise IndexError(
                    f"Cannot find `{key}` in table index `{self.table._index}`")
        elif hasattr(key, "dtype"):
            if key.dtype.kind in "SU":
                if self.table._multiple_row_selections:
                    mask[self.table._get_names_indices(key)] = True
                else:
                    return self.table._get_names_indices(key) # preserve key order
            else:
                mask[key] = True
        elif isinstance(key, list):
            if len(key) > 0 and isinstance(key[0], str):
                if self.table._multiple_row_selections:
                    mask[self.table._get_names_indices(key)] = True
                else:
                    return self.table._get_names_indices(key) # preserve key order
            else:
                mask[key] = True
        elif isinstance(key, slice):
            ia = key.start
            ib = key.stop
            ic = key.step
            if isinstance(ia, str) or isinstance(ib, str):
                if ic is None:
                    ic = self.table._index
                if ia is not None:
                    ia = self.table._get_name_indices(ia, ic)[0]
                if ib is not None:
                    ib = self.table._get_name_indices(ib, ic)[-1] + 1
                mask[ia:ib] = True
            elif isinstance(ic, str):
                col = self.table._data[ic]
                if ia is None and ib is None:
                    mask |= True
                elif ia is not None and ib is None:
                    mask |= col <= ib
                elif ib is not None and ia is None:
                    mask |= col >= ia
                else:
                    mask |= (col >= ia) & (col <= ib)
            else:
                mask[ia:ib:ic] = True
        elif isinstance(key, tuple):
            if self.table._multiple_row_selections:
                mask = self[key[0]]
                if len(key) > 1:
                    mask &= self[key[1:]]
            else:
                return self.__getitem__(list(key))

        return mask

class _RowView:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, rows):
        restore_multiple_row_selections = self.table._multiple_row_selections
        self.table._multiple_row_selections = True
        try:
            out = self.table._get_rows_cols(rows, None)
        except Exception as err:
            self.table._multiple_row_selections = restore_multiple_row_selections
            raise err
        self.table._multiple_row_selections = restore_multiple_row_selections
        return out

class _ColView:
    def __init__(self, table):
        self.table = table

    def __getitem__(self, cols):
        return self.table._get_rows_cols(None, cols, force_table=True)

    @property
    def names(self):
        return self.table._col_names

    def __repr__(self):
        return "<" + " ".join(self.table._col_names) + ">"



class _View:
    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __getitem__(self, k):
        return self.data[k][self.index]

    def __len__(self):
        k = list(self.data)[0]
        return len(self.data[k])

    def get(self, k, default=None):
        if k == '__tracebackhide__': # to avoid issues in ipython
            return None
        return self.data.get(k, default)[self.index]

    def __repr__(self):
        return f"<{self.table._nrows} rows>"

    def __iter__(self):
        return iter(self.table._data[self.table._index])




class RDMTable:

    _multiple_row_selections = True
    _error_on_row_not_found = False

    def __init__(
        self,
        data,
        col_names=None,
        index="name",
        header=None,
        count_sep="##",
        offset_sep="%%",
        index_cache=None,
    ):
        self._data = data

        self._col_names = list(data.keys()) if col_names is None else col_names
        for kk in self._col_names:
            vv = data[kk]
            if not hasattr(vv, 'dtype'):
                raise ValueError(f"Column `{kk}` is not a numpy array")
        self._index = index
        self._count_sep = count_sep
        self._offset_sep = offset_sep
        self.mask = Mask(self)
        self._index_cache = index_cache
        self._regex_flags = re.IGNORECASE
        nrows = set(len(self._data[cc]) for cc in self._col_names)
        assert len(nrows) == 1
        self._nrows = nrows.pop()
        self.rows = _RowView(self)
        self.cols = _ColView(self)

    def to_pandas(self, index=None, columns=None):

        if columns is None:
            columns = self._col_names

        import pandas as pd
        df = pd.DataFrame(self._data, columns=self._col_names)
        if index is not None:
            df.set_index(index, inplace=True)
        return df

    def _get_index(self):
        if self._index in self._data:
            return self._data[self._index]
        else:
            raise ValueError(f"Cannot find `{self._index}` in table")

    def _get_index_cache(self):
        if self._index_cache is None:
            col = self._get_index()
            dct = {}
            count = {}
            col = self._get_index()
            for ii, nn in enumerate(col):
                cc = count.get(nn, -1) + 1
                dct[(nn, cc)] = ii
                count[nn] = cc
            self._index_cache = dct
        return self._index_cache

    def _split_name_count_offset(self, name):
        ss = name.split(self._count_sep)
        name = ss[0]
        count = None if len(ss) == 1 else int(ss[1])
        ss = name.split(self._offset_sep)
        name = ss[0]
        offset = 0 if len(ss) == 1 else int(ss[1])
        return name, count, offset

    def _get_name_mask(self, name, col):
        name, count, offset = self._split_name_count_offset(name)
        if col == self._index:
            tryout = self._get_index_cache().get((name, count))
            if tryout is not None:
                mask = np.zeros(self._nrows, dtype=bool)
                mask[tryout] = True
                return mask
        col = self._data[col]
        regex = re.compile(name, re.IGNORECASE)
        it = (regex.fullmatch(rr) is not None for rr in col)
        mask = np.fromiter(it, count=self._nrows, dtype=bool)
        if count is not None:
            idx = np.where(mask)[0][count]
            mask = np.zeros(self._nrows, dtype=bool)
            mask[idx] = True
        return mask

    def _get_name_indices(self, name, col):
        name, count, offset = self._split_name_count_offset(name)
        if col == self._index:
            idx = self._get_index_cache().get((name, count))
            if idx is not None:
                return [idx + offset]
        regex = re.compile(name, self._regex_flags)
        lst = []
        cnt = -1
        for ii, nn in enumerate(self._data[col]):
            if regex.fullmatch(nn):
                cnt += 1
                if count is None or count == cnt:
                    lst.append(ii)
        return np.array(lst, dtype=int) + offset

    def _get_name_index(self, name, col):
        return self._get_name_indices(name, col)[0]

    def _get_names_indices(self, names):
        dct = self._get_index_cache()
        lst = []
        for name in names:
            name, count, offset = self._split_name_count_offset(name)
            if self._error_on_row_not_found and name not in self[self._index]:
                raise IndexError(f"Cannot find `{name}` in table index `{self._index}`")
            if count is None:
                count = 0
            lst.append(dct[(name, count)] + offset)
        return np.array(lst, dtype=int)

    def __getattr__(self, key):
        if key in self._data:
            return self._data[key]
        else:
            raise AttributeError(f"Cannot find `{key}` in table")

    def __len__(self):
        return len(self._data)

    def keys(self, exclude_columns=False):
        if exclude_columns:
            return [kk for kk in self._data.keys() if kk not in self._col_names]
        return self._data.keys()

    def values(self):
        return self._data.values()

    def items(self):
        return self._data.items()

    def __dir__(self):
        return super().__dir__() + list(self._data.keys())

    def __iter__(self):
        return self._data.__iter__()

    def __contains__(self):
        return self._data.__contains__()

    def __setitem__(self, key, val):
        if len(val) != self._nrows:
            raise ValueError("Wrong number of rows")
        self._col_names.append(key)
        self._data[key] = val
        if key == self._index:
            self._index_cache = None

    def __delitem__(self, key, val):
        self._col_names.remove(key)
        del self._data[key]

    def __setattr__(self, key, val):
        if key == '_data':
            super().__setattr__(key, val)
            return
        if key == "_index":
            super().__setattr__(key, val)
            self._index_cache = None
            return
        if key in self._data:
            if key in self._col_names:
                self._data[key][:] = val
                return
            else:
                self._data[key] = val
                return
        super().__setattr__(key, val)

    def __repr__(self):
        n = self._nrows
        c = len(self._col_names)
        ns = "s" if n != 1 else ""
        cs = "s" if c != 1 else ""
        out = [f"{self.__class__.__name__}: {n} row{ns}, {c} col{cs}"]
        if self._nrows < 10000:
            show = self.show(output=str)
            out.append(show)
        return "\n".join(out)

    def __getitem__(self, args):
        if type(args) is str and args in self._data:
            return self._data[args]
        if type(args) is tuple:  # multiple args
            if len(args) == 0:
                cols = None
                rows = None
            elif len(args) == 1:
                cols = args[0]
                rows = None
            elif len(args) == 2:
                cols = args[0]
                rows = args[1]
            else:
                if self._multiple_row_selections:
                    cols = args[0]
                    rows = args[1:]
                else:
                    raise ValueError(
                        "Too many indices or keys. Expected usage is "
                        "`table[col]` or `table[row, col]` or "
                        "`table[[row1, row2, ...], [col1, col2, ...]]`")
        else:  # one arg
            cols = args
            rows = None
        return self._get_rows_cols(rows, cols)

    def _get_view_col_list(self, rows, cols):
        # select rows
        if rows is None:
            view = self._data
        else:
            row_index = self.mask[rows]
            view = _View(self._data, row_index)

        # select cols
        if cols is None or cols == slice(None, None, None):
            col_list = self._col_names
        elif type(cols) is str:
            col_list = cols.split()
        else:
            col_list = list(cols)

        return view, col_list

    def _get_rows_cols(self, rows, cols, force_table=False):
        view, col_list = self._get_view_col_list(rows, cols)

        # return data
        if len(col_list) == 1 and not force_table:
            try:
                cc = eval(col_list[0], gblmath, view)
            except NameError:
                raise KeyError(
                    f"Column `{col_list[0]}` could not be found or "
                    "is not a valid expression"
                )
            if len(cc) == 1:
                return cc[0]  # scalar
            else:
                return cc  # array
        else:
            if self._index not in col_list:
                col_list.insert(0, self._index)
            data = {}
            for cc in col_list:
                try:
                    data[cc] = eval(cc, gblmath, view)
                except NameError:
                    raise KeyError(
                        f"Column `{cc}` could not be found or "
                        "is not a valid expression"
                    )
            for kk in self.keys(exclude_columns=True):
                data[kk] = self._data[kk]
            return self.__class__(
                data, index=self._index, count_sep=self._count_sep,
                col_names=col_list
            )  # table

    def show(
        self,
        rows=None,
        cols=None,
        maxrows=20,
        maxwidth=80,
        output=None,
        digits=6,
        fixed="g",
        header=True,
    ):
        view, col_list = self._get_view_col_list(rows, cols)

        # add index
        if self._index not in col_list:
            col_list.insert(0, self._index)

        cut = -1
        viewrows = len(view)
        if maxrows is not None and output is None and viewrows > maxrows:
            cut = maxrows // 2

        data = []
        width = 0
        # maxwidth=10000000 if maxwidth is None else maxwidth
        fmt = []
        header_line = []
        for cc in col_list:
            if cc in view:
                coldata = view[cc]
            else:
                coldata = eval(cc, gblmath, view)
            if cut > 0:
                coldata = np.r_[coldata[:cut], coldata[cut:]]
            coltype = coldata.dtype.kind
            col = _to_str(coldata, digits, fixed)
            colwidth = int(col.dtype.str[2:])
            if len(cc) > colwidth:
                colwidth = len(cc)
            width += colwidth + 1
            if width < maxwidth:
                if coltype in "SU":
                    fmt.append("%%-%ds" % (colwidth))
                else:
                    fmt.append("%%%ds" % colwidth)
                header_line.append(fmt[-1] % cc)
                data.append(col)
            else:
                header_line.append('...')
                break


        result = []
        if header:
             result.append(" ".join(header_line))
        for ii in range(len(col)):
            row = " ".join([ff % col[ii] for ff, col in zip(fmt, data)])
            result.append(row)
            if ii == cut:
                result.append("...")
        result = "\n".join(result)
        if output is None:
            print(result)
        elif output is str:
            return result
        elif hasattr(output, "write"):
            output.write(result)
        else:
            output = pathlib.Path(output)
            with open(output, "w") as fh:
                fh.write(result)

class Table(RDMTable):

    _multiple_row_selections = False
    _error_on_row_not_found = True

