from .mav import Mav


class Mj(Mav):
    def candScore(self, scores):
        """This formula will always give numbers within 0.5 of the raw median.
        Unfortunately, with 5 grade levels, these will tend to be within 0.1 of
        the raw median, leaving scores further from the integers mostly unused.
        This is only a problem aesthetically.

        For now, only works correctly for odd nvot

        tests:
            >>> Mj().candScore([1,2,3,4,5])
            3
            >>> Mj().candScore([1,2,3,3,5])
            2.7
            >>> Mj().candScore([1,3,3,3,5])
            3
            >>> Mj().candScore([1,3,3,4,5])
            3.3
            >>> Mj().candScore([1,3,3,3,3])
            2.9
            >>> Mj().candScore([3] * 24 + [1])
            2.98
            >>> Mj().candScore([3] * 24 + [4])
            3.02
            >>> Mj().candScore([3] * 13 + [4] * 12)
            3.46
            """
        scores = sorted(scores)
        nvot = len(scores)
        lo = hi = mid = nvot // 2
        base = scores[mid]
        while (hi < nvot and scores[hi] == base):
            hi += 1
        while (lo >= 0 and scores[lo] == base):
            lo -= 1

        if (hi-mid) == (mid-lo):
            return base
        elif (hi-mid) < (mid-lo):
            return base + 0.5 - (hi-mid) / nvot
        else:
            return base - 0.5 + (mid-lo) / nvot
