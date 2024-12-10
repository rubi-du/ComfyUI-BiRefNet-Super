from __future__ import annotations

from typing import List

from .base import BaseSession

sessions_class: List[type[BaseSession]] = []
sessions_names: List[str] = []

from ..birefnet_general import BiRefNetSessionGeneral

sessions_class.append(BiRefNetSessionGeneral)
sessions_names.append(BiRefNetSessionGeneral.name())

from ..birefnet_general_lite import BiRefNetSessionGeneralLite

sessions_class.append(BiRefNetSessionGeneralLite)
sessions_names.append(BiRefNetSessionGeneralLite.name())

from ..birefnet_portrait import BiRefNetSessionPortrait

sessions_class.append(BiRefNetSessionPortrait)
sessions_names.append(BiRefNetSessionPortrait.name())

from ..birefnet_dis import BiRefNetSessionDIS

sessions_class.append(BiRefNetSessionDIS)
sessions_names.append(BiRefNetSessionDIS.name())

from ..birefnet_hrsod import BiRefNetSessionHRSOD

sessions_class.append(BiRefNetSessionHRSOD)
sessions_names.append(BiRefNetSessionHRSOD.name())

from ..birefnet_cod import BiRefNetSessionCOD

sessions_class.append(BiRefNetSessionCOD)
sessions_names.append(BiRefNetSessionCOD.name())

from ..birefnet_massive import BiRefNetSessionMassive

sessions_class.append(BiRefNetSessionMassive)
sessions_names.append(BiRefNetSessionMassive.name())
